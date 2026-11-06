"""CPU/GPU 스모크용 합성 수학 데이터. 실제 CoVA 로더와 동일한 컬럼(prompts, task, ref)을 낸다."""
from __future__ import annotations
import json, random

SYSTEM = ("Please reason step by step, and put your final answer within \\boxed{}. "
          "You may sketch intermediate figures with visual tokens <I...> if it helps.")


def make_synthetic(n: int, seed: int = 0, **_) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a, b = rng.randint(2, 30), rng.randint(2, 30)
            q, ans = f"Compute {a} * {b}.", str(a * b)
        elif kind == 1:
            a, b = rng.randint(3, 12), rng.randint(3, 12)
            q, ans = f"A right triangle has legs {a} and {b}. What is its area?", str(a * b / 2)
        else:
            r = rng.randint(1, 9)
            q, ans = f"A circle has radius {r}. What is its area? Give an exact answer.", f"{r*r}\\pi"
        rows.append({"prompts": f"{SYSTEM}\n{q}", "task": "math", "ref": json.dumps({"answer": ans})})
    return rows
