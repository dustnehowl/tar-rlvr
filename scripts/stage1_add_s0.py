"""prompts.jsonl 에 GT 첫 이미지의 S0 (729 토큰) 인코딩을 추가한다 (같은 300 프롬프트 유지). CPU.
사용: /mnt/work/venv311/bin/python scripts/stage1_add_s0.py [--prompts logs/stage1/prompts.jsonl]
"""
import argparse, hashlib, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="logs/stage1/prompts.jsonl")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.prompts)]
need = {r["qid"] for r in rows}
from datasets import load_dataset
imgs = {}
for t in ("math_vr", "mathnet"):
    for r in load_dataset("tsunghanwu/cova", t, split="train"):
        q = hashlib.md5(r["question"].encode()).hexdigest()[:12]
        if q in need and q not in imgs:
            imgs[q] = r["images"][0]
from tar_rlvr.vis.tatok import TATok, to_token_string
tk = TATok()
cache = {}
for i, r in enumerate(rows):
    if r["qid"] not in cache:
        cache[r["qid"]] = to_token_string(*tk.encode(Image.open(io.BytesIO(imgs[r["qid"]])).convert("RGB"), 0))
    r["gt_tokens_s0"] = cache[r["qid"]]
    print(f"[s0] {i + 1}/{len(rows)}", flush=True)
for r in rows:
    r["corrupt_tokens_s0"] = cache[r["corrupt_qid"]]
tmp = a.prompts + ".tmp"
with open(tmp, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(tmp, a.prompts)
print("[s0] done")
