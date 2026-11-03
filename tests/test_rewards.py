import json
from tar_rlvr.rewards import correctness_reward, format_reward, make_visual_budget_reward
from tar_rlvr.rewards.verifiers import grid_exact, grid_cellwise, sudoku_valid, graph_coloring_valid, exact_match, numeric_match
from tar_rlvr.rewards.parsing import parse_grid, extract_answer


def test_parse_grid_formats():
    assert parse_grid("[[0,1],[2,3]]") == [[0, 1], [2, 3]]
    assert parse_grid("0 1\n2 3") == [[0, 1], [2, 3]]
    assert parse_grid("0 1\n2") is None


def test_grid():
    ref = {"answer": [[1, 2], [3, 4]]}
    assert grid_exact("[[1,2],[3,4]]", ref) == 1.0
    assert grid_exact("[[1,2],[3,5]]", ref) == 0.0
    assert grid_cellwise("[[1,2],[3,5]]", ref) == 0.75
    assert grid_cellwise("[[1,2]]", ref) == 0.0


def test_sudoku_4x4():
    puzzle = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    sol = "[[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]]"
    assert sudoku_valid(sol, {"puzzle": puzzle}) == 1.0
    # 다른 유효 해도 1.0 (정답 비교가 아닌 제약 검증)
    sol2 = "[[1,3,2,4],[2,4,1,3],[4,1,3,2],[3,2,4,1]]"
    assert sudoku_valid(sol2, {"puzzle": puzzle}) == 1.0
    # 주어진 칸 변경 → 0
    bad = "[[2,1,3,4],[3,4,1,2],[1,2,4,3],[4,3,2,1]]"
    assert sudoku_valid(bad, {"puzzle": puzzle}) == 0.0


def test_graph_coloring():
    ref = {"edges": [("A", "B"), ("B", "C")], "k": 2}
    assert graph_coloring_valid("A:1, B:2, C:1", ref) == 1.0
    assert graph_coloring_valid('{"A":1,"B":2,"C":1}', ref) == 1.0
    assert graph_coloring_valid("A:1, B:1, C:2", ref) == 0.0   # 인접 충돌
    assert graph_coloring_valid("A:1, B:2, C:3", ref) == 0.0   # k 초과
    assert graph_coloring_valid("A:1, B:2", ref) == 0.0        # 노드 누락


def test_generic():
    assert exact_match("  Grandmother. ", {"answer": "grandmother"}) == 1.0
    assert numeric_match("so the total is 1,234.", {"answer": 1234}) == 1.0
    assert numeric_match("no number", {"answer": 1}) == 0.0


def test_reward_fns_end_to_end():
    comps = [
        "<I1><I2><I3> reasoning... <answer>[[1,2],[3,4]]</answer>",
        "<answer>[[0,0],[0,0]]</answer>",
        "no tags at all",
    ]
    cols = {"task": ["grid"] * 3, "ref": [json.dumps({"answer": [[1, 2], [3, 4]]})] * 3}
    assert correctness_reward(None, comps, **cols) == [1.0, 0.0, 0.0]
    assert format_reward(None, comps, **cols) == [1.0, 0.5, 0.0]
    vb = make_visual_budget_reward(max_tokens_per_image=2)
    r = vb(None, comps, **cols)
    assert 0.0 < r[0] < 1.0 and r[1] == 0.0


def test_extract_last_answer():
    assert extract_answer("<answer>a</answer> wait <answer>b</answer>") == "b"
