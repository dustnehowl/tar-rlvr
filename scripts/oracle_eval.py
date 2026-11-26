"""Stage 1: oracle on demand (TPU, 학습 없음). 기본 Tar-7B (LoRA 없음) 로 4 조건을 같은 300 프롬프트에서 비교한다.

조건
  text_only      : 중립 지시문 (sketch 문장 없음) + <im_start> 금지.
  model_drawn    : sketch 지시문, 개입 없음. 모델이 그린 그림 뒤에 bridge 를 붙여 이어쓴다 (r06 과 같은 이어쓰기).
  oracle         : sketch 지시문. 모델이 <im_start> 를 내는 순간 멈추고, 모델의 그림 대신
                   GT 첫 이미지 토큰 (<S2> + 81 + <im_end>) + bridge 를 넣고 이어서 생성.
  corrupt        : oracle 과 같되 다른 문제의 GT 첫 이미지를 넣는다.

세 그림 조건은 <im_start> 까지의 앞부분(prefix)을 **같은 샘플끼리 공유**한다: prefix 를 한 번 생성하고
(<im_start> 또는 EOS 에서 멈춤) 거기서 세 갈래로 나눈다. 그래서 그림 조건끼리는 샘플 단위로 정확히 짝지어진다.
<im_start> 를 안 낸 샘플은 세 조건 모두 똑같다 (개입할 지점이 없음).
이어쓰기에서 또 <im_start> 를 내면 같은 규칙을 다시 적용 (oracle/corrupt 는 같은 GT 를 다시 넣음), 그림 최대 --max-images 장.
그 뒤에는 <im_start> 를 금지한다. completion 예산은 그림·bridge 토큰 포함 768 (r06 eval 과 같음).

--force-image (2026-11-26, Stage 1b): 그림 조건 전부 completion 첫 토큰에 <im_start> 를 강제 (prefix 생성 없음).
  maze 에서 기본 모델이 <im_start> 를 거의 안 내서 on-demand 로는 개입이 일어나지 않기 때문. 다른 out 디렉터리에 쓸 것.

추가 조건 (2026-11-26): model_drawn_S0 = <im_start> 뒤에 <S0> 를 강제하고 729 토큰을 모델이 그림,
  oracle_S0 = GT 를 S0 (729 토큰) 으로 넣음. 예산 1536 (729 토큰 그림 + bridge 뒤에도 ≥ 600 토큰 남음).
  model_drawn 의 그림 상한은 760 토큰 (첫 실행의 96 은 샘플링에서 S1/S0 그림을 잘라 답을 못 내게 했다).

출력: <out>/records.jsonl — (kind=greedy|sampled, qid, sample, cond) 마다 한 줄. 같은 kind 를 다시 돌리면 건너뛴다.
사용: /mnt/work/venv311/bin/python scripts/oracle_eval.py --out logs/stage1/base [--limit 16 --kinds greedy]
"""
import argparse, copy, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml

IM_START, IM_END = "<im_start>", "<im_end>"
SEQ = {0: 729, 1: 169, 2: 81}
# cond → (mode, arg). model: 모델이 그림 (arg = 강제 스케일 또는 None), inject: prompts.jsonl 의 토큰 열 이름
COND_SPECS = {
    "model_drawn": ("model", None),
    "model_drawn_S0": ("model", 0),
    "oracle_S2": ("inject", "gt_tokens"),
    "oracle_S0": ("inject", "gt_tokens_s0"),
    "corrupt": ("inject", "corrupt_tokens"),
    "corrupt_S0": ("inject", "corrupt_tokens_s0"),    # Stage 1b: oracle_S0 의 대조군
    "oracle": ("inject", "gt_tokens"),          # 첫 실행 (logs/stage1/base) 의 이름 = oracle_S2
}
DEFAULT_CONDS = "text_only,model_drawn,model_drawn_S0,oracle_S2,oracle_S0,corrupt"


def patch_keep_stop_token():
    """Tunix sampler 는 정지 토큰을 출력에서 잘라낸다 → 어떤 토큰에서 멈췄는지 알 수 있게 정지 토큰까지 남긴다."""
    from tunix.generate import sampler as sampler_mod
    u = sampler_mod.utils
    if getattr(u, "_stage1_patched", False):
        return
    orig = u.np_find_first_eos_idx

    def keep(ids, eos_id):
        i = orig(ids, eos_id)
        return i + 1 if i < len(ids) else i
    u.np_find_first_eos_idx = keep
    u._stage1_patched = True


class Gen:
    def __init__(self, cfg, batch: int, prompt_len: int, kv: int):
        import jax
        from tunix.generate import sampler as sampler_mod
        from tunix.generate.tokenizer_adapter import TokenizerAdapter
        from tunix.utils import compat
        from tar_rlvr.train.grpo import build_mesh, build_model_and_tokenizer
        patch_keep_stop_token()
        self.mesh = build_mesh(cfg["mesh"])
        self._ctx = compat.set_mesh(self.mesh)
        self._ctx.__enter__()
        model, tok = build_model_and_tokenizer(cfg, self.mesh, with_lora=False)
        self.tok = tok
        mc = model.config
        self.sampler = sampler_mod.Sampler(model, TokenizerAdapter(tok), sampler_mod.CacheConfig(
            cache_size=kv, num_layers=mc.num_layers, num_kv_heads=mc.num_kv_heads, head_dim=mc.head_dim))
        self.B, self.P = batch, prompt_len
        self.id = {s: tok.convert_tokens_to_ids(s) for s in (IM_START, IM_END)}
        self.eos = tok.eos_token_id
        self.calls = 0

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def run(self, prompts, steps, stops, forbid=(), greedy=True, temperature=0.8, seed=0):
        """→ [(text_without_stop, n_tokens_incl_stop, stop)] ; stop ∈ {'eos','<im_start>','<im_end>','length'}."""
        stop_ids = [self.eos if s == "eos" else self.id[s] for s in stops]
        name = {v: k for k, v in zip(stops, stop_ids)}
        res = []
        for s in range(0, len(prompts), self.B):
            chunk = list(prompts[s:s + self.B])
            n = len(chunk)
            chunk += [chunk[-1]] * (self.B - n)
            for p in chunk:
                assert self.n_tokens(p) <= self.P, "prompt longer than --prompt-len"
            out = self.sampler(input_strings=chunk, max_generation_steps=steps, max_prompt_length=self.P,
                               eos_tokens=stop_ids, forbidden_tokens=list(forbid) or None,
                               temperature=1.0 if greedy else temperature, top_p=None if greedy else 1.0,
                               top_k=None, seed=None if greedy else seed * 100003 + s, pad_output=False)
            self.calls += 1
            for i in range(n):
                ids = [int(t) for t in np.asarray(out.tokens[i])]
                stop = name.get(ids[-1]) if ids else None
                body = ids[:-1] if stop else ids
                res.append((self.tok.decode(body), len(ids), stop or "length"))
        return res


def score(text, row):
    from tar_rlvr.rewards.parsing import extract_answer
    from tar_rlvr.rewards.verifiers import VERIFIERS
    from tar_rlvr.rewards.cova_tasks import SCORERS
    if row["task"] in SCORERS:                         # Stage 1b: maze / clutrr / prosqa (태스크별 답 추출)
        ok, has, _ = SCORERS[row["task"]](text, json.loads(row["ref"]))
        return bool(ok), bool(has)
    a = extract_answer(text)
    ok = VERIFIERS[row["task"]](a, json.loads(row["ref"])) >= 1 if a is not None else False
    return bool(ok), a is not None


def run_kind(gen, rows, kind, G, a, bridge, write):
    greedy = kind == "greedy"
    items = [(r, j) for r in rows for j in range(1 if greedy else G)]
    budget = a.budget
    t0 = time.time()
    seed = 1 if greedy else 7

    # ---- text_only
    if "text_only" in a.conds.split(","):
        outs = gen.run([r["prompt_neutral"] for r, _ in items], budget, ["eos"], forbid=[gen.id[IM_START]],
                       greedy=greedy, temperature=a.temperature, seed=seed)
        recs = []
        for (r, j), (txt, n, stop) in zip(items, outs):
            ok, has = score(txt, r)
            recs.append(dict(kind=kind, qid=r["qid"], sample=j, cond="text_only", completion=txt, n_tokens=n, stop=stop,
                       n_images=0, emitted=False, first_pos=None, correct=ok, has_answer=has, truncated=stop == "length"))
        write(recs)
        print(f"[stage1] {kind} text_only done ({time.time() - t0:.0f}s)", flush=True)

    # ---- shared prefix (sketch prompt) — <im_start> 또는 EOS 에서 멈춤
    # prefix 는 디스크에 저장해 재사용 → 나중에 추가하는 조건도 정확히 같은 prefix 에서 갈라진다
    pre_path = f"{a.out}/prefix_{kind}.jsonl"
    if a.force_image:                                 # Stage 1b: completion 맨 앞에 그림을 강제 (prefix 없음)
        pre = [("", 1, IM_START) for _ in items]
        print(f"[stage1] {kind} prefix: forced <im_start> at token 0", flush=True)
    elif os.path.exists(pre_path):
        pre = [tuple(json.loads(l)) for l in open(pre_path)]
        assert len(pre) == len(items), (len(pre), len(items))
        print(f"[stage1] {kind} prefix loaded from {pre_path}", flush=True)
    else:
        pre = gen.run([r["prompt_sketch"] for r, _ in items], budget, ["eos", IM_START],
                      greedy=greedy, temperature=a.temperature, seed=seed + 1)
        with open(pre_path + ".tmp", "w") as f:
            for x in pre:
                f.write(json.dumps(list(x), ensure_ascii=False) + "\n")
        os.replace(pre_path + ".tmp", pre_path)
    base = []
    for (r, j), (txt, n, stop) in zip(items, pre):
        emitted = stop == IM_START
        base.append(dict(r=r, j=j, comp=txt + (IM_START if emitted else ""), used=n, n_img=0,
                         pending=emitted, done=not emitted, stop=stop, emitted=emitted,
                         first_pos=(n - 1) if emitted else None, prefix_tokens=n, events=[]))
    print(f"[stage1] {kind} prefix: emitted {sum(b['emitted'] for b in base)}/{len(base)} ({time.time() - t0:.0f}s)", flush=True)

    bridge_n = gen.n_tokens(bridge)
    for cond in a.conds.split(","):
        if cond == "text_only":
            continue
        mode, arg = COND_SPECS[cond]
        st = copy.deepcopy(base)
        rnd = 0
        while True:
            # 1) 그림 채우기
            pend = [s for s in st if s["pending"] and not s["done"]]
            if mode == "model" and pend:
                if arg is not None:                       # 스케일 토큰 강제 (r05 의 강제 prefix 처럼)
                    for s in pend:
                        s["comp"] += f"<S{arg}>"
                        s["used"] += 1
                outs = gen.run([s["r"]["prompt_sketch"] + s["comp"] for s in pend], a.image_steps, [IM_END, "eos"],
                               greedy=greedy, temperature=a.temperature, seed=seed + 10 + rnd)
                for s, (txt, n, stop) in zip(pend, outs):
                    codes = re.findall(r"<I\d+>", txt)
                    if arg is not None and stop == "length" and len(codes) >= SEQ[arg]:
                        # 강제 스케일: 시각 토큰 SEQ 개를 다 그렸는데 안 닫으면 거기서 닫는다
                        txt, n, stop = "".join(codes[: SEQ[arg]]), SEQ[arg] + 1, IM_END
                        s["events"].append("image_forced_close")
                    s["used"] += n
                    if stop != IM_END:                    # 그림이 안 닫힘 → r06 처럼 이어쓰지 않음
                        s.update(done=True, pending=False, stop=f"image_{stop}")
                        s["comp"] += txt
                        s["events"].append("image_unclosed")
                    else:
                        s["comp"] += txt + IM_END
                        m = re.search(r"<im_start>(?:<S(\d)>)?((?:<I\d+>)*)<im_end>$", s["comp"])
                        s["events"].append(f"image_scale{m.group(1) if m else '?'}_codes{len(re.findall('<I', m.group(2))) if m else -1}")
            elif pend:
                for s in pend:
                    tokens = s["r"][arg]
                    assert tokens.startswith(IM_START)
                    s["comp"] += tokens[len(IM_START):]
                    s["used"] += gen.n_tokens(tokens) - 1
            for s in pend:
                if s["done"]:
                    continue
                s["n_img"] += 1
                s["pending"] = False
                s["comp"] += bridge
                s["used"] += bridge_n
                s["events"].append("image")
                if s["used"] >= budget:
                    s.update(done=True, stop="length")
            # 2) 텍스트 이어쓰기
            act = [s for s in st if not s["done"] and not s["pending"]]
            if not act:
                break
            for capped in (False, True):
                grp = [s for s in act if (s["n_img"] >= a.max_images) == capped]
                if not grp:
                    continue
                outs = gen.run([s["r"]["prompt_sketch"] + s["comp"] for s in grp], budget, ["eos", IM_START],
                               forbid=[gen.id[IM_START]] if capped else (),
                               greedy=greedy, temperature=a.temperature, seed=seed + 20 + rnd)
                for s, (txt, n, stop) in zip(grp, outs):
                    remaining = budget - s["used"]
                    if n > remaining:                      # 예산 초과 → 자른다
                        ids = gen.tok.encode(txt, add_special_tokens=False)[:remaining]
                        txt, n, stop = gen.tok.decode(ids), remaining, "length"
                    s["used"] += n
                    s["comp"] += txt
                    s["stop"] = stop
                    if stop == IM_START:
                        s["comp"] += IM_START
                        s["pending"] = True
                        if s["used"] >= budget:
                            s.update(done=True, stop="length")
                    else:
                        s["done"] = True
            rnd += 1
        recs = []
        for s in st:
            ok, has = score(s["comp"], s["r"])
            recs.append(dict(kind=kind, qid=s["r"]["qid"], sample=s["j"], cond=cond, completion=s["comp"], n_tokens=s["used"],
                       stop=s["stop"], n_images=s["n_img"], emitted=s["emitted"], first_pos=s["first_pos"],
                       prefix_tokens=s["prefix_tokens"], correct=ok, has_answer=has, truncated=s["stop"] == "length",
                       events=s["events"]))
        write(recs)
        print(f"[stage1] {kind} {cond} done, rounds {rnd} ({time.time() - t0:.0f}s, sampler calls {gen.calls})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tpu_v6e_7b.yaml")
    ap.add_argument("--prompts", default="logs/stage1/prompts.jsonl")
    ap.add_argument("--out", default="logs/stage1/base")
    ap.add_argument("--kinds", default="greedy,sampled")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1536)
    ap.add_argument("--image-steps", type=int, default=760)   # S0 (729) 도 닫힐 수 있게
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--conds", default=DEFAULT_CONDS)
    ap.add_argument("--force-image", action="store_true",
                    help="모델이 <im_start> 를 낼 때까지 기다리지 않고 completion 첫 토큰을 <im_start> 로 강제 (Stage 1b maze: greedy 0/200 이 그림을 안 그림)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    bridge = cfg["grpo"]["continuation"]["bridge"]
    rows = [json.loads(l) for l in open(a.prompts)][: a.limit]
    os.makedirs(a.out, exist_ok=True)
    meta = dict(vars(a), bridge=bridge, model=cfg["model"]["hf_id"], lora=False, n_prompts=len(rows))
    json.dump(meta, open(f"{a.out}/run_config.json", "w"), indent=1)
    path = f"{a.out}/records.jsonl"
    done = set()                                       # (kind, cond) 단위로 이어하기
    if os.path.exists(path):
        cnt = {}
        for r in map(json.loads, open(path)):
            cnt[(r["kind"], r["cond"])] = cnt.get((r["kind"], r["cond"]), 0) + 1
        done = {kc for kc, n in cnt.items() if n == len(rows) * (1 if kc[0] == "greedy" else a.G)}
    gen = Gen(cfg, a.batch, a.prompt_len, kv=a.prompt_len + a.budget)
    for kind in a.kinds.split(","):
        todo = [c for c in a.conds.split(",") if (kind, c) not in done]
        if not todo:
            print(f"[stage1] {kind} already complete — skip"); continue
        a_k = argparse.Namespace(**dict(vars(a), conds=",".join(todo)))

        def flush(recs):                               # 조건 단위로 원자적으로 기록
            with open(path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[stage1] {kind}/{recs[0]['cond']}: wrote {len(recs)} records", flush=True)
        run_kind(gen, rows, kind, a.G, a_k, bridge, flush)


if __name__ == "__main__":
    main()
