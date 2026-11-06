"""브리지 접두어 탐색 (학습 없음).

1) 문제 N개 × G개 샘플링해서 그림(<im_end>)으로 끝난 completion 을 모은다
2) 각 브리지 후보를 붙여 이어 생성하고, 결과가 '질문 복창' / '풀이' / '답 도달' 중 뭔지 센다
사용: uv run python scripts/probe_bridge.py --config configs/gpu_7b.yaml --n-problems 24 --g 4
"""
from __future__ import annotations
import argparse, dataclasses, re, yaml, random
import jax
from tunix.utils import compat
from tunix.generate import sampler as sampler_lib
from tar_rlvr.model import tar as tar_lib
from tar_rlvr.train.grpo import build_mesh, format_prompt, CHAT_SYSTEM
from tar_rlvr.data.cova import load_cova
from tar_rlvr.rewards.parsing import extract_answer

BRIDGES = [
    "\nBased on the sketch, ",
    "\nFrom the sketch, I can see that",
    "\nLooking at the figure above, let me work through this step by step.\n",
    "\nUsing the diagram I just drew, the key observation is that",
    "\nThe sketch above shows the setup. Now solving:\n",
    "\nNow, reasoning step by step with the help of the sketch:\n\nStep 1:",
]
_QUESTIONY = re.compile(r"^\s*(what|which|how|find|determine|calculate|given)\b|\?\s*$", re.I)


def classify(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if "<im_start>" in t:
        return "redraw"
    if extract_answer(t) is not None and ("\\boxed" in t or re.search(r"answer\s*(is|:)", t, re.I)):
        return "answer"
    if _QUESTIONY.search(t.split("\n")[0]) and len(t) < 200:
        return "restate"
    return "reasoning" if len(t) > 80 else "short"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu_7b.yaml")
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--max-first", type=int, default=160)
    ap.add_argument("--max-cont", type=int, default=256)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mesh = build_mesh(cfg["mesh"])
    with compat.set_mesh(mesh):
        from huggingface_hub import snapshot_download
        path = snapshot_download(cfg["model"]["hf_id"], ignore_patterns=["*.pth"])
        model, mcfg = tar_lib.load_tar7b(path, mesh, remat=False)
        tok = tar_lib.load_tokenizer(path)
        smp = sampler_lib.Sampler(model, tok, sampler_lib.CacheConfig(
            cache_size=2048, num_layers=mcfg.num_layers, num_kv_heads=mcfg.num_kv_heads, head_dim=mcfg.head_dim))
        d = dict(cfg["data"]); d["n_samples"] = a.n_problems; d["seed"] = 1
        rows = load_cova(d)
        prompts = [format_prompt(tok, r["prompts"]) for r in rows]

        def gen(ps, max_steps, seed):
            out = smp(input_strings=ps, max_generation_steps=max_steps, max_prompt_length=cfg["grpo"]["max_prompt_length"],
                      temperature=0.8, top_p=1.0, seed=seed, pad_output=True)
            return list(out.text)

        # 1) 1차: 그림으로 끝난 것 수집
        firsts = []
        for gi in range(a.g):
            texts = gen(prompts, a.max_first, seed=100 + gi)
            for p, t in zip(prompts, texts):
                if "<im_start>" in t and "<im_end>" in t:
                    firsts.append((p, t.split("<im_end>")[0] + "<im_end>"))
        print(f"\n1차 생성 {len(prompts) * a.g}개 중 그림으로 끝난 것 {len(firsts)}개 "
              f"({len(firsts) / (len(prompts) * a.g):.0%})\n")
        if not firsts:
            print("그림 샘플이 없음 — 프롬프트 힌트/온도 확인"); return
        random.seed(0); firsts = random.sample(firsts, min(len(firsts), 24))

        # 2) 브리지별 이어쓰기
        print(f"{'bridge':<70} {'answer':>6} {'reason':>6} {'restate':>7} {'redraw':>6} {'empty':>5} {'short':>5}")
        examples = {}
        for b in BRIDGES:
            conts = gen([p + f + b for p, f in firsts], a.max_cont, seed=7)
            cls = [classify(c) for c in conts]
            cnt = {k: cls.count(k) for k in ("answer", "reasoning", "restate", "redraw", "empty", "short")}
            label = repr(b)[:68]
            print(f"{label:<70} {cnt['answer']:>6} {cnt['reasoning']:>6} {cnt['restate']:>7} {cnt['redraw']:>6} {cnt['empty']:>5} {cnt['short']:>5}")
            examples[b] = [(k, c) for k, c in zip(cls, conts)]
        print("\n" + "=" * 90 + "\n각 브리지별 예시 2개 (answer/reasoning 우선)\n")
        for b, ex in examples.items():
            print(f"--- {b!r}")
            ex.sort(key=lambda kc: {"answer": 0, "reasoning": 1}.get(kc[0], 2))
            for k, c in ex[:2]:
                print(f"  [{k}] {c.strip()[:300]!r}")
            print()


if __name__ == "__main__":
    main()
