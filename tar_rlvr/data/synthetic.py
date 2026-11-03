"""CPU 스모크용 합성 데이터. 실제 CoVA 스키마와 동일한 컬럼(prompts, task, ref)을 낸다."""
from __future__ import annotations
import json, random

SYSTEM = ("You may think with visual tokens <I...> to sketch intermediate states. "
          "Give the final answer inside <answer></answer>.")


def _grid(rng):
    g = [[rng.randint(0, 3) for _ in range(3)] for _ in range(3)]
    t = [row[::-1] for row in g]  # 규칙: 좌우 반전
    return {"prompts": f"{SYSTEM}\nFlip this grid horizontally: {json.dumps(g)}",
            "task": "grid", "ref": json.dumps({"answer": t})}


def _sudoku(rng):
    base = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
    puz = [[v if rng.random() < 0.5 else 0 for v in r] for r in base]
    return {"prompts": f"{SYSTEM}\nSolve this 4x4 sudoku (0 = empty): {json.dumps(puz)}",
            "task": "sudoku", "ref": json.dumps({"puzzle": puz})}


def _coloring(rng):
    edges = [["A", "B"], ["B", "C"], ["C", "D"]]
    return {"prompts": f"{SYSTEM}\nColor this graph with 2 colors. Edges: {edges}. Answer as A:1, B:2, ...",
            "task": "graph_coloring", "ref": json.dumps({"edges": edges, "k": 2})}


GEN = {"grid": _grid, "sudoku": _sudoku, "graph_coloring": _coloring}


def make_synthetic(n: int, tasks: list[str], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [GEN[tasks[i % len(tasks)]](rng) for i in range(n)]
