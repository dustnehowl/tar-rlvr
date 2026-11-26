"""Probe: r06 이 bridge 없이도 '그림 뒤에 이어 쓰기' 를 배웠나? (TPU, 학습 없음)

base Tar-7B 와 r06 (LoRA, 체크포인트 step 176) 을 held-out 100 (logs/stage1/prompts.jsonl 의 split=heldout,
r06 eval 과 같은 프롬프트, sketch 지시문) 에서 이어쓰기 없이 (bridge 주입 없음) 생성한다.
greedy 1 + T=0.8 샘플 8, 예산 768, EOS 에서만 멈춤.

completion 마다: <im_start> 냈나, 첫 <im_end> 바로 뒤가 EOS / 텍스트 / 또 <im_start> 인가,
텍스트면 몇 토큰이고 bridge 같은 말 ("Now, reasoning", "Step 1") 로 시작하나, 답 도달·정답.

teacher forcing: 두 모델의 completion 에서 그림이 닫힌 문맥 (프롬프트 + completion 의 첫 <im_end> 까지) 을 모아
**두 모델 모두** 같은 문맥에서 다음 토큰 분포를 잰다 → P(EOS), P(<im_start>), P(bridge 첫 토큰),
P(bridge 첫 4 토큰 "\\nNow, reasoning" 연속).

모델 하나에 LoRA 를 붙여 쓴다: LoRA B 초기값 = 0 → base 와 같다. 체크포인트 복원 → r06. 초기 LoRA 를 저장해 두고 되돌린다.
출력: <out>/records.jsonl, <out>/tf.jsonl, <out>/report.md
사용: /mnt/work/venv311/bin/python scripts/probe_continuation.py --out logs/probe_continuation
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import yaml

IM_START, IM_END = "<im_start>", "<im_end>"
BRIDGE_RE = re.compile(r"^\s*Now,?\s+reasoning", re.I)
STEP1_RE = re.compile(r"\bStep\s*1\b", re.I)


def classify(text, stop, ids_after):
    """첫 <im_end> 뒤 → (after, n_text_tokens, bridge_like, step1_early)."""
    k = text.find(IM_END)
    rest = text[k + len(IM_END):]
    if rest.startswith(IM_START):
        return "image", 0, False, False
    if rest.strip() == "":
        return ("eos" if stop == "eos" else "length"), 0, False, False
    head = rest.lstrip()
    return "text", ids_after, bool(BRIDGE_RE.match(head)), bool(STEP1_RE.search(head[:200]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tpu_v6e_7b.yaml")
    ap.add_argument("--prompts", default="logs/stage1/prompts.jsonl")
    ap.add_argument("--ckpt-root", default="/mnt/work/ckpt/r06-sketch-bridge/actor")
    ap.add_argument("--step", type=int, default=176)
    ap.add_argument("--out", default="logs/probe_continuation")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--budget", type=int, default=768)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--tf-batch", type=int, default=4)
    ap.add_argument("--tf-len", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    bridge = cfg["grpo"]["continuation"]["bridge"]
    rows = [r for r in map(json.loads, open(a.prompts)) if r["split"] == "heldout"][: a.limit]
    assert a.limit or len(rows) == 100, len(rows)

    import jax, jax.numpy as jnp
    from flax import nnx
    from tunix.utils import compat
    from tunix.sft.checkpoint_manager import CheckpointManager
    from tar_rlvr.train.grpo import build_mesh, build_model_and_tokenizer
    from oracle_eval import Gen, score

    mesh = build_mesh(cfg["mesh"])
    with compat.set_mesh(mesh):
        model, tok = build_model_and_tokenizer(cfg, mesh, with_lora=True)
    init_lora = jax.tree.map(lambda x: jnp.array(x, copy=True), nnx.state(model, nnx.LoRAParam))   # 초기값 사본 (B=0 → base)
    b_norm = lambda: float(np.sqrt(sum(float(jnp.sum(jnp.square(v.astype(jnp.float32))))
                                       for p, v in jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.LoRAParam))
                                       if "lora_b" in jax.tree_util.keystr(p))))
    gen = Gen(cfg, a.batch, prompt_len=1024, kv=1024 + a.budget, model_tok=(model, tok))
    eos, ims, ime = gen.eos, gen.id[IM_START], gen.id[IM_END]
    bridge_ids = tok.encode(bridge, add_special_tokens=False)
    print(f"[probe] eos={eos} im_start={ims} im_end={ime} bridge[:4]={bridge_ids[:4]} {[tok.decode([t]) for t in bridge_ids[:4]]}", flush=True)

    def set_lora(which):
        if which == "base":
            nnx.update(model, init_lora)
        else:
            CheckpointManager(root_directory=a.ckpt_root).maybe_restore(model, step=a.step, restore_only_lora_params=True)
        gen.rebuild_sampler()
        print(f"[probe] LoRA = {which}: ||B||_F = {b_norm():.4f}", flush=True)

    # ------------------------------------------------------------ 생성
    recs = []
    for which in ("base", "r06"):
        set_lora(which)
        t0 = time.time()
        for kind, G in (("greedy", 1), ("sampled", a.G)):
            items = [(r, j) for r in rows for j in range(G)]
            outs = gen.run([r["prompt_sketch"] for r, _ in items], a.budget, ["eos"], greedy=kind == "greedy",
                           temperature=0.8, seed=11, return_ids=True)
            for (r, j), (txt, n, stop, ids) in zip(items, outs):
                ok, has = score(txt, r)
                emitted = IM_START in txt
                closed = emitted and IM_END in txt
                rec = dict(model=which, kind=kind, qid=r["qid"], sample=j, n_tokens=n, stop=stop, emitted=emitted,
                           closed=closed, n_images=txt.count(IM_START), correct=ok, has_answer=has,
                           completion=txt, ids=ids)
                if closed:
                    k = ids.index(ime)
                    after = len(ids) - (k + 1) - (1 if stop == "eos" else 0)
                    rec["after"], rec["after_tokens"], rec["bridge_like"], rec["step1_early"] = classify(txt, stop, after)
                    rec["ctx_ids"] = ids[: k + 1]
                elif emitted:
                    rec["after"] = "unclosed"
                recs.append(rec)
            print(f"[probe] {which} {kind}: {len(outs)} done ({time.time() - t0:.0f}s)", flush=True)
    with open(f"{a.out}/records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("ids",)}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ teacher forcing (같은 문맥, 두 모델)
    P = {r["qid"]: r for r in rows}
    ctxs = []
    for r in recs:
        if "ctx_ids" in r:
            p_ids = tok.encode(P[r["qid"]]["prompt_sketch"], add_special_tokens=False)
            full = p_ids + r["ctx_ids"]
            if len(full) + 4 <= a.tf_len:
                ctxs.append(dict(src=r["model"], kind=r["kind"], qid=r["qid"], sample=r["sample"], ids=full))
    print(f"[probe] TF contexts: {len(ctxs)}", flush=True)
    b4 = bridge_ids[:4]
    L = a.tf_len

    @nnx.jit
    def tf_step(m, toks, last):
        B = toks.shape[0]
        pos = jnp.broadcast_to(jnp.arange(L)[None], (B, L))
        mask = jnp.broadcast_to(jnp.tril(jnp.ones((L, L), bool))[None], (B, L, L))
        logits, _ = m(toks, pos, None, mask)
        idx = last[:, None] + jnp.arange(4)[None]                                  # 문맥 끝 + bridge 3 토큰 위치
        lg = jnp.take_along_axis(logits, idx[:, :, None], axis=1).astype(jnp.float32)   # [B,4,V]
        return jax.nn.log_softmax(lg, axis=-1)

    tf = []
    for which in ("r06", "base"):
        set_lora(which)
        for s in range(0, len(ctxs), a.tf_batch):
            ch = ctxs[s:s + a.tf_batch]
            n = len(ch)
            ch = ch + [ch[-1]] * (a.tf_batch - n)
            toks = np.zeros((a.tf_batch, L), np.int32)
            last = np.zeros((a.tf_batch,), np.int32)
            for i, c in enumerate(ch):
                seq = c["ids"] + b4
                toks[i, :len(seq)] = seq
                last[i] = len(c["ids"]) - 1
            lp = np.asarray(tf_step(model, jnp.asarray(toks), jnp.asarray(last)))
            for i in range(n):
                c = ch[i]
                l0 = lp[i, 0]
                tf.append(dict(scorer=which, src=c["src"], kind=c["kind"], qid=c["qid"], sample=c["sample"],
                               p_eos=float(np.exp(l0[eos])), p_im_start=float(np.exp(l0[ims])),
                               p_bridge1=float(np.exp(l0[b4[0]])),
                               p_bridge4=float(np.exp(sum(lp[i, t, b4[t]] for t in range(4)))),
                               top1=int(l0.argmax())))
        print(f"[probe] TF {which} done", flush=True)
    with open(f"{a.out}/tf.jsonl", "w") as f:
        for t in tf:
            f.write(json.dumps(t) + "\n")
    report(recs, tf, tok, a.out)


def report(recs, tf, tok, out):
    import collections
    md = ["# Probe: 그림 뒤에 스스로 이어 쓰나 (bridge 없음) — base vs r06 (step 176)\n",
          "held-out 100, sketch 지시문, 이어쓰기 없음, 예산 768. sampled = 100 × 8, T=0.8.\n"]
    md.append("| | " + " | ".join(f"{m} {k}" for m in ("base", "r06") for k in ("greedy", "sampled")) + " |")
    md.append("|---|" + "---|" * 4)

    def col(fn):
        vals = []
        for m in ("base", "r06"):
            for k in ("greedy", "sampled"):
                rr = [r for r in recs if r["model"] == m and r["kind"] == k]
                vals.append(fn(rr))
        return vals

    pct = lambda x, n: f"{100 * x / max(n, 1):.1f}% ({x}/{n})"
    rows = [
        ("<im_start> 냄", lambda rr: pct(sum(r["emitted"] for r in rr), len(rr))),
        ("그림 닫힘 (<im_end>)", lambda rr: pct(sum(r["closed"] for r in rr), len(rr))),
    ]
    for lab, key in (("↳ <im_end> 뒤 EOS", "eos"), ("↳ <im_end> 뒤 텍스트", "text"), ("↳ <im_end> 뒤 또 <im_start>", "image"),
                     ("↳ <im_end> 뒤 예산 소진", "length")):
        rows.append((lab, lambda rr, key=key: pct(sum(r.get("after") == key for r in rr), sum(r["closed"] for r in rr))))
    rows += [
        ("그림 안 닫힘", lambda rr: pct(sum(r.get("after") == "unclosed" for r in rr), sum(r["emitted"] for r in rr))),
        ("텍스트 뒤따름: 토큰 중앙값", lambda rr: str(int(np.median([r["after_tokens"] for r in rr if r.get("after") == "text"])) if any(r.get("after") == "text" for r in rr) else "-")),
        ("텍스트가 'Now, reasoning' 로 시작", lambda rr: pct(sum(r.get("bridge_like", False) for r in rr), sum(r.get("after") == "text" for r in rr))),
        ("텍스트 앞 200자에 'Step 1'", lambda rr: pct(sum(r.get("step1_early", False) for r in rr), sum(r.get("after") == "text" for r in rr))),
        ("답 도달 (전체)", lambda rr: pct(sum(r["has_answer"] for r in rr), len(rr))),
        ("정답 (전체)", lambda rr: pct(sum(r["correct"] for r in rr), len(rr))),
        ("정답 (그림 낸 completion)", lambda rr: pct(sum(r["correct"] for r in rr if r["emitted"]), sum(r["emitted"] for r in rr))),
        ("정답 (그림 안 낸 completion)", lambda rr: pct(sum(r["correct"] for r in rr if not r["emitted"]), sum(not r["emitted"] for r in rr))),
    ]
    for lab, fn in rows:
        md.append(f"| {lab} | " + " | ".join(col(fn)) + " |")

    md.append("\n## <im_end> 직후 다음 토큰 확률 (teacher forcing, 같은 문맥에서 두 모델)\n")
    md.append("문맥 = 두 모델의 completion 에서 그림이 닫힌 지점까지 (프롬프트 포함). 평균 확률, 괄호는 중앙값.\n")
    md.append("| 문맥 출처 | n | scorer | P(EOS) | P(<im_start>) | P(bridge 첫 토큰) | P(bridge 4토큰 연속) | 1 − P(EOS) − P(<im_start>) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for src in ("base", "r06", "all"):
        for sc in ("base", "r06"):
            tt = [t for t in tf if t["scorer"] == sc and (src == "all" or t["src"] == src)]
            if not tt:
                continue
            f = lambda k: f"{np.mean([t[k] for t in tt]):.3f} ({np.median([t[k] for t in tt]):.3f})"
            rest = [1 - t["p_eos"] - t["p_im_start"] for t in tt]
            md.append(f"| {src} | {len(tt)} | {sc} | {f('p_eos')} | {f('p_im_start')} | {f('p_bridge1')} | {f('p_bridge4')} | {np.mean(rest):.3f} |")
    # 짝지은 차이 (같은 문맥)
    key = lambda t: (t["src"], t["kind"], t["qid"], t["sample"])
    B = {key(t): t for t in tf if t["scorer"] == "base"}
    R = {key(t): t for t in tf if t["scorer"] == "r06"}
    ks = sorted(set(B) & set(R))
    if ks:
        rng = np.random.default_rng(0)
        md.append("\n짝지은 차이 r06 − base (같은 문맥, 프롬프트 단위 bootstrap 95% CI):\n")
        for k_ in ("p_eos", "p_im_start", "p_bridge1", "p_bridge4"):
            d = collections.defaultdict(list)
            for k in ks:
                d[k[2]].append(R[k][k_] - B[k][k_])
            qs = list(d)
            means = [np.mean([np.mean(d[q]) for q in rng.choice(qs, len(qs))]) for _ in range(2000)]
            md.append(f"- {k_}: {np.mean([np.mean(d[q]) for q in qs]):+.3f} [{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}] (n 문맥 {len(ks)}, 프롬프트 {len(qs)})")
        top = collections.Counter(tok.decode([t["top1"]]) for t in tf if t["scorer"] == "r06")
        topb = collections.Counter(tok.decode([t["top1"]]) for t in tf if t["scorer"] == "base")
        md.append(f"\n가장 확률 높은 다음 토큰 — base: {topb.most_common(5)} / r06: {top.most_common(5)}")
    open(f"{out}/report.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
