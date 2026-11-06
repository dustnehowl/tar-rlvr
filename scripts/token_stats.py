# scripts/token_stats.py
"""프롬프트 / 정답 풀이 토큰 길이 분포. max_prompt_length 와 풀이-길이 필터 기준을 정하는 용도."""
import json, numpy as np, yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tar_rlvr.model.tar import load_tokenizer
from tar_rlvr.data.cova import _ref, _keep
from tar_rlvr.data.synthetic import SYSTEM

cfg = yaml.safe_load(open("configs/gpu_7b.yaml"))
tok = load_tokenizer(snapshot_download(cfg["model"]["hf_id"], ignore_patterns=["*.pth"]))
P, R = [], []
for name in ["math_vr", "mathnet"]:
    ds = load_dataset("tsunghanwu/cova", name, split="train").remove_columns(["images"])
    for r in ds:
        if _keep(_ref(r), 60, r["question"]): continue
        P.append(len(tok.encode(SYSTEM + "\n" + r["question"])))
        sol = r["interleaved_reasoning"]
        sol = sol if isinstance(sol, str) else json.dumps(sol)
        R.append(len(tok.encode(sol)))
for lab, x in [("prompt", P), ("reference solution", R)]:
    x = np.array(x); q = np.percentile(x, [50, 90, 95, 99])
    print(f"{lab:>20}: n={len(x)} mean={x.mean():.0f} p50={q[0]:.0f} p90={q[1]:.0f} p95={q[2]:.0f} p99={q[3]:.0f} max={x.max()}")
R = np.array(R)
for t in [384, 512, 768, 1024]:
    print(f"  solution ≤ {t} tokens: {(R <= t).mean():.0%}")

# --- 저장: qid → 참조 풀이 토큰 수 (cova.py 의 solution_len_file 필터용)
import hashlib
out = {}
for name in ["math_vr", "mathnet"]:
    ds = load_dataset("tsunghanwu/cova", name, split="train").remove_columns(["images"])
    for r in ds:
        sol = r["interleaved_reasoning"]; sol = sol if isinstance(sol, str) else json.dumps(sol)
        out[hashlib.md5(r["question"].encode()).hexdigest()[:12]] = len(tok.encode(sol))
json.dump(out, open("data/solution_len.json", "w"))
print("saved data/solution_len.json")
