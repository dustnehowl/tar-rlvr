"""CPU/GPU 스모크용 합성 수학 데이터. 실제 CoVA 로더와 동일한 컬럼(prompts, task, ref)을 낸다."""
from __future__ import annotations
import json, random

SYSTEM_NEUTRAL = "Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_SKETCH = SYSTEM_NEUTRAL + " You may sketch intermediate figures with visual tokens <I...> if it helps."
SYSTEM = SYSTEM_SKETCH          # run1~run4 에서 쓴 지시문 (기본값 유지)


def instruction(kind: str = "sketch") -> str:
    """data.instruction: "sketch" = 그림을 유도하는 문장 포함 (run1~4), "neutral" = 일반 RLVR 지시문만."""
    return {"sketch": SYSTEM_SKETCH, "neutral": SYSTEM_NEUTRAL}[kind]


def make_synthetic(n: int, seed: int = 0, instruction_kind: str = "sketch", **_) -> list[dict]:
    system = instruction(instruction_kind)
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
        rows.append({"prompts": f"{system}\n{q}", "task": "math", "ref": json.dumps({"answer": ans})})
    return rows
