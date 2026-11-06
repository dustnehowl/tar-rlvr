"""기본 모델 pass-rate 스캔 (학습 없음). 문제 N개 × K회 샘플링 → 문제별 맞춘 횟수.
학습 셋에서 전부-맞음/전부-틀림 문제를 빼기 위한 것 (GRPO 는 그룹 내 대비가 있어야 신호가 생김).
롤아웃 규칙은 학습과 동일하게: 그림으로 끝나면 bridge 붙여 이어쓰기.

사용: uv run python scripts/scan_passrate.py --n 400 --k 8 --out data/passrate.json
이후 configs 의 data 에  passrate_file: data/passrate.json  추가.
"""
from __future__ import annotations
import argparse, json, os, time, yaml
import numpy as np
from tunix.utils import compat
from tunix.generate import sampler as sampler_lib
from tar_rlvr.model import tar as tar_lib
from tar_rlvr.train.grpo import build_mesh, format_prompt
from tar_rlvr.data.cova import load_cova
from tar_rlvr.rewards import correctness_reward
from tar_rlvr.rewards.parsing import extract_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu_7b.yaml")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="data/passrate.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); g = cfg["grpo"]
    bridge = g.get("continuation", {}).get("bridge", "\nNow, reasoning step by step with the help of the sketch:\n\nStep 1:")
    max_comp = g["max_completion_length"]
    mesh = build_mesh(cfg["mesh"])
    with compat.set_mesh(mesh):
        from huggingface_hub import snapshot_download
        path = snapshot_download(cfg["model"]["hf_id"], ignore_patterns=["*.pth"])
        model, mcfg = tar_lib.load_tar7b(path, mesh, remat=False)
        tok = tar_lib.load_tokenizer(path)
        smp = sampler_lib.Sampler(model, tok, sampler_lib.CacheConfig(
            cache_size=2048, num_layers=mcfg.num_layers, num_kv_heads=mcfg.num_kv_heads, head_dim=mcfg.head_dim))
        d = dict(cfg["data"]); d["n_samples"] = a.n; d["passrate_file"] = None
        rows = load_cova(d)
        prompts = [format_prompt(tok, r["prompts"]) for r in rows]

        def gen(ps, max_steps, seed):
            out = smp(input_strings=ps, max_generation_steps=max_steps, max_prompt_length=g["max_prompt_length"],
                      temperature=g.get("temperature", 0.8), top_p=1.0, seed=seed, pad_output=True)
            return list(out.text)

        stats = {r["qid"]: {"pass": 0, "k": a.k, "vis_n": 0, "vis_pass": 0} for r in rows}
        t0 = time.time()
        for ki in range(a.k):
            for b in range(0, len(rows), a.batch):
                sl = slice(b, b + a.batch)
                texts = gen(prompts[sl], max_comp, seed=1000 * ki + b)
                # 이어쓰기 (학습 때 continuation.py 와 같은 규칙, 텍스트 수준)
                idx = [i for i, t in enumerate(texts)
                       if "<im_start>" in t and "<im_end>" in t and extract_answer(t) is None]
                if idx:
                    conts = gen([prompts[sl][i] + texts[i].split("<im_end>")[0] + "<im_end>" + bridge for i in idx],
                                max_comp // 2, seed=7 + ki)
                    for i, c in zip(idx, conts):
                        texts[i] = texts[i].split("<im_end>")[0] + "<im_end>" + bridge + c
                rs = correctness_reward(None, texts, task=["math"] * len(texts),
                                        ref=[r["ref"] for r in rows[sl]])
                for r, t, sc in zip(rows[sl], texts, rs):
                    st = stats[r["qid"]]
                    st["pass"] += int(sc >= 1.0)
                    if "<im_start>" in t:
                        st["vis_n"] += 1; st["vis_pass"] += int(sc >= 1.0)
                done = ki * len(rows) + b + len(texts)
                print(f"  {done}/{a.k * len(rows)}  {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(stats, open(a.out, "w"), indent=1)

    p = np.array([s["pass"] for s in stats.values()])
    print("\n=== pass@k 분포 (문제별 맞춘 횟수 / k) ===")
    for v in range(a.k + 1):
        print(f"  {v}/{a.k}: {int((p == v).sum()):4d} 문제")
    useful = int(((p >= 1) & (p <= a.k - 1)).sum())
    print(f"\n평균 정답률 {p.mean() / a.k:.3f}   학습에 쓸 수 있는(1~{a.k - 1}) 문제: {useful}/{len(p)} ({useful / len(p):.0%})")
    vn = sum(s["vis_n"] for s in stats.values()); vp = sum(s["vis_pass"] for s in stats.values())
    tn = a.k * len(p) - vn; tp = int(p.sum()) - vp
    print(f"시각 샘플 정답률 {vp}/{vn} = {vp / max(1, vn):.3f}   텍스트 샘플 정답률 {tp}/{tn} = {tp / max(1, tn):.3f}")
    print(f"\n저장: {a.out}  → configs 의 data 에  passrate_file: {a.out}  추가")


if __name__ == "__main__":
    main()
