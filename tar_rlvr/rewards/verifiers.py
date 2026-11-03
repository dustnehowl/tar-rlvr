"""Task별 verifier. 전부 순수 파이썬 → pytest로 하드웨어 없이 검증.

각 verifier: (pred_answer: str, ref: dict) -> float in [0, 1]
ref 는 데이터셋 row 에서 뽑은 정답/제약 정보 (task별 키가 다름).
"""
from __future__ import annotations
import re
from .parsing import parse_grid


# ---------- 그리드 계열: ARC, Maze, Table 등 ----------
def grid_exact(pred: str, ref: dict) -> float:
    g = parse_grid(pred)
    t = parse_grid(ref["answer"]) if isinstance(ref["answer"], str) else ref["answer"]
    return float(g is not None and g == t)


def grid_cellwise(pred: str, ref: dict) -> float:
    """부분 점수 (shaping용). 크기 틀리면 0."""
    g = parse_grid(pred)
    t = parse_grid(ref["answer"]) if isinstance(ref["answer"], str) else ref["answer"]
    if g is None or t is None or len(g) != len(t) or any(len(a) != len(b) for a, b in zip(g, t)):
        return 0.0
    cells = sum(len(r) for r in t)
    ok = sum(int(a == b) for ra, rb in zip(g, t) for a, b in zip(ra, rb))
    return ok / cells


# ---------- Sudoku: 정답 비교 대신 '유효한 해인지' 검증 (RLVR의 장점) ----------
def sudoku_valid(pred: str, ref: dict) -> float:
    g = parse_grid(pred)
    puzzle = parse_grid(ref["puzzle"]) if isinstance(ref["puzzle"], str) else ref["puzzle"]
    if g is None or puzzle is None:
        return 0.0
    n = len(g)
    box = int(n ** 0.5)
    if box * box != n or any(len(r) != n for r in g):
        return 0.0
    # 주어진 칸 보존
    for i in range(n):
        for j in range(n):
            if puzzle[i][j] != 0 and puzzle[i][j] != g[i][j]:
                return 0.0
    full = set(range(1, n + 1))
    for i in range(n):
        if set(g[i]) != full or set(g[r][i] for r in range(n)) != full:
            return 0.0
    for bi in range(0, n, box):
        for bj in range(0, n, box):
            if {g[i][j] for i in range(bi, bi + box) for j in range(bj, bj + box)} != full:
                return 0.0
    return 1.0


# ---------- Graph coloring: k색 이하 + 인접 노드 색 다름 ----------
def graph_coloring_valid(pred: str, ref: dict) -> float:
    """pred 예: 'A:1, B:2, C:1' 또는 json {"A":1,...}; ref: edges=[(u,v),...], k=int"""
    import json
    try:
        colors = json.loads(pred)
    except Exception:
        colors = {}
        for m in re.finditer(r"([A-Za-z0-9_]+)\s*[:=]\s*([A-Za-z0-9_]+)", pred):
            colors[m.group(1)] = m.group(2)
    if not colors:
        return 0.0
    nodes = {str(n) for e in ref["edges"] for n in e}
    if not nodes.issubset({str(k) for k in colors}):
        return 0.0
    colors = {str(k): str(v) for k, v in colors.items()}
    if len(set(colors[n] for n in nodes)) > ref["k"]:
        return 0.0
    for u, v in ref["edges"]:
        if colors[str(u)] == colors[str(v)]:
            return 0.0
    return 1.0


# ---------- 일반: 정규화 후 문자열/숫자 비교 (CLUTRR, ProsQA, 수학 등) ----------
def _norm(s: str) -> str:
    s = s.strip().lower().rstrip(".")
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, ref: dict) -> float:
    return float(_norm(pred) == _norm(str(ref["answer"])))


def numeric_match(pred: str, ref: dict, tol: float = 1e-6) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    if not nums:
        return 0.0
    try:
        return float(abs(float(nums[-1]) - float(ref["answer"])) <= tol)
    except Exception:
        return 0.0


# task 이름 → verifier. CoVA 17개 task 이름은 스키마 확인 후 채우기.
VERIFIERS = {
    "grid": grid_exact,
    "arc": grid_exact,
    "maze": grid_exact,
    "sudoku": sudoku_valid,
    "graph_coloring": graph_coloring_valid,
    "exact": exact_match,
    "numeric": numeric_match,
}
