"""Stage 1 준비 (CPU): 300 프롬프트 + GT 첫 이미지의 TA-Tok 토큰 + corrupt 짝 + leaky 판정 → logs/stage1/prompts.jsonl

- 풀: configs/tpu_v6e_7b.yaml 의 data 설정 그대로 (passrate·풀이 길이 필터 = r06 과 같은 548 문제).
- held-out 100: build_dataset 과 같은 방식 (qid 정렬 → eval.seed 셔플 → 앞 100). 나머지에서 200 개 (seed 0).
- GT: CoVA `images[0]` (= interleaved_reasoning 의 첫 <img>) 를 S2 (81 토큰) 로 인코딩. Tar 가 스스로 그릴 때 쓰는 형식과 같다.
- corrupt: 300 개 안에서 고정점 없는 순열 (seed 0) — 다른 문제의 GT 첫 이미지.
- leaky (휴리스틱, 텍스트 기반): 정답 값이 trace 의 첫 <img> 앞 텍스트 또는 그림 직후 설명(첫 문단)에 나오면 leaky.
  그림 자체에 정답이 적혀 있는지는 텍스트로 알 수 없다 → 보고서에서 표본을 눈으로 확인.

사용: /mnt/work/venv311/bin/python scripts/stage1_prep.py
"""
import argparse, hashlib, io, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from PIL import Image


def canon(s: str) -> str:
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\(text|mathrm|operatorname)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\^\{?\\circ\}?|°|\\degree", "deg", s)
    return re.sub(r"[\s$\\{}]", "", s).lower()


def leaky(trace: str, gold_value: str):
    k = trace.find("<img>")
    if k < 0:
        return True, "no_img_in_trace"
    before = trace[:k]
    after = trace[k + 5:]
    after = after.split("\n\n", 2)
    after = "\n\n".join(after[:2])[:600]                      # 그림 직후 설명 문단
    if "boxed" in before:
        return True, "answer_before_first_image"
    g = canon(gold_value)
    if not g:
        return False, ""
    for name, w in (("before", before), ("figure_description", after)):
        cw = canon(w)
        if len(g) >= 3 and g in cw:
            return True, f"gold_in_{name}"
        if len(g) < 3 and re.search(r"(=|\bis\b)\s*\$?\s*" + re.escape(gold_value.strip()) + r"(?![\d.])", w):
            return True, f"gold_in_{name}"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tpu_v6e_7b.yaml")
    ap.add_argument("--n-extra", type=int, default=200)
    ap.add_argument("--out", default="logs/stage1/prompts.jsonl")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    from tar_rlvr.data.cova import load_cova
    from tar_rlvr.data.synthetic import SYSTEM_SKETCH, SYSTEM_NEUTRAL
    from tar_rlvr.model.tar import load_tokenizer
    from tar_rlvr.train.grpo import format_prompt
    rows = load_cova(dict(cfg["data"], instruction="sketch"))
    ev = cfg["eval"]
    order = sorted(rows, key=lambda r: str(r["qid"]))
    random.Random(ev.get("seed", 4242)).shuffle(order)
    held, rest = order[: ev["n_prompts"]], order[ev["n_prompts"]:]
    extra = random.Random(0).sample(sorted(rest, key=lambda r: r["qid"]), a.n_extra)
    print(f"[prep] pool {len(rows)} → held-out {len(held)} + extra {len(extra)}")

    from datasets import load_dataset
    cova = {}
    for t in cfg["data"]["tasks"]:
        for r in load_dataset(cfg["data"]["hf_id"], t, split="train"):
            q = hashlib.md5(r["question"].encode()).hexdigest()[:12]
            cova.setdefault(q, dict(r, task_name=t))
    tok = load_tokenizer(cfg["model"]["tokenizer_path"])
    from tar_rlvr.vis.tatok import TATok, to_token_string
    tk = TATok()
    out = []
    for split, rs in (("heldout", held), ("extra", extra)):
        for r in rs:
            c = cova[r["qid"]]
            ref = json.loads(r["ref"])
            assert c["images"], r["qid"]
            img = Image.open(io.BytesIO(c["images"][0])).convert("RGB")
            scale, codes = tk.encode(img, 2)
            gold_value = ref["options"][ref["choice"]] if ref.get("options") else ref["answer"]
            lk, why = leaky(c["interleaved_reasoning"], gold_value)
            k = c["interleaved_reasoning"].find("<img>")
            out.append({"qid": r["qid"], "split": split, "subset": c["task_name"], "task": r["task"], "ref": r["ref"],
                        "is_mc": bool(ref.get("options")), "question": c["question"],
                        "prompt_sketch": format_prompt(tok, f"{SYSTEM_SKETCH}\n{c['question']}"),
                        "prompt_neutral": format_prompt(tok, f"{SYSTEM_NEUTRAL}\n{c['question']}"),
                        "gt_tokens": to_token_string(scale, codes), "n_gt_images": len(c["images"]),
                        "trace_first_img_frac": k / max(1, len(c["interleaved_reasoning"])),
                        "leaky": lk, "leaky_reason": why})
            print(f"[prep] {len(out)}/{len(held) + len(extra)} {r['qid']} leaky={lk} {why}", flush=True)
    perm = list(range(len(out)))
    rng = random.Random(0)
    while True:
        rng.shuffle(perm)
        if all(i != p for i, p in enumerate(perm)):
            break
    for i, p in enumerate(perm):
        out[i]["corrupt_qid"] = out[p]["qid"]
        out[i]["corrupt_tokens"] = out[p]["gt_tokens"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n_l = sum(o["leaky"] for o in out)
    print(f"[prep] wrote {len(out)} → {a.out}; leaky {n_l}; MC {sum(o['is_mc'] for o in out)}")


if __name__ == "__main__":
    main()
