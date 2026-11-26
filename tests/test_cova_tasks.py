"""maze / clutrr / prosqa 채점 (Stage 1b). 하드웨어 불필요."""
import json
from tar_rlvr.rewards.cova_tasks import score_maze, score_clutrr, score_prosqa, prosqa_ref, maze_from_question
from tar_rlvr.rewards import correctness_reward

RAW = "S000\n1101\n000E"
REF = {"answer": "S220\n1121\n002E", "raw_maze": RAW}


def test_maze_gold_and_formats():
    assert score_maze("S220\n1121\n002E", REF)[0]
    assert score_maze("Final:\n\\boxed{S220\\\\1121\\\\0022}", REF)[0]
    assert score_maze("```\nS 2 2 0\n1 1 2 1\n0 0 2 E\n```", REF)[0]
    assert score_maze("2220\n1121\n0022", REF)[0]          # S/E 를 2 로 바꿔도 됨
    assert score_maze("S222\n1121\n002E", REF)[0]          # (0,3) 은 경로에 붙어 있음 → 유효


def test_maze_invalid():
    assert not score_maze("UNSOLVABLE", REF)[0]
    assert not score_maze(RAW, REF)[0]                      # 경로 없음
    assert not score_maze("S220\n1121\n000E", REF)[0]       # 끊긴 경로
    assert not score_maze("S220\n1121\n202E", REF)[0]       # 따로 떨어진 2
    assert not score_maze("S220\n1221\n002E", REF)[0]       # 벽을 2 로 바꿈
    assert score_maze("no grid here", REF) == (False, False, None)


def test_maze_unsolvable_and_last_wins():
    ref = {"answer": "UNSOLVABLE", "raw_maze": "S1\n1E"}
    assert score_maze("the maze is UNSOLVABLE", ref)[0]
    assert not score_maze("S2\n12", ref)[0]
    assert score_maze("maybe unsolvable?\nS220\n1121\n002E", REF)[0]   # 나중에 나온 격자가 답


def test_maze_question_parse():
    q = "You are given ...\n\nMaze:\nS000\n1101\n000E\n\nFind a path from S to E."
    assert maze_from_question(q) == RAW


def test_clutrr():
    assert score_clutrr("so she is his \\boxed{grandmother}", {"answer": "grandmother"})[0]
    assert not score_clutrr("\\boxed{mother}", {"answer": "grandmother"})[0]
    assert not score_clutrr("\\boxed{granddaughter}", {"answer": "daughter"})[0]
    assert score_clutrr("\\boxed{niece}", {"answer": "neice"})[0]
    assert score_clutrr("Answer: son-in-law", {"answer": "son-in-law"})[0]
    assert not score_clutrr("Answer: son-in-law", {"answer": "son"})[0]


def test_prosqa():
    ref = prosqa_ref("Every a is b. Tom is a fompus. Is Tom a worpus or quimpus?", "Tom is a worpus.")
    assert ref["options"] == ["worpus", "quimpus"] and ref["gold"] == "worpus"
    assert score_prosqa("\\boxed{Tom is a worpus}", ref)[0]
    assert not score_prosqa("\\boxed{Tom is a quimpus}", ref)[0]
    assert score_prosqa("\\boxed{Tom is a worpus, not a quimpus}", ref)[0]
    assert not score_prosqa("\\boxed{worpus or quimpus}", ref)[0]


def test_correctness_reward_dispatch():
    r = correctness_reward(None, ["S220\n1121\n002E", "\\boxed{aunt}"], task=["maze", "clutrr"],
                           ref=[json.dumps(REF), json.dumps({"answer": "aunt"})])
    assert r == [1.0, 1.0]


def test_maze_boxed_takes_precedence():
    ref = {"answer": "UNSOLVABLE", "raw_maze": "S1\n1E"}
    assert not score_maze("If we don't reach E, print UNSOLVABLE.\nFinal:\n\\boxed{2202000002}", ref)[0]
    assert score_maze("\\boxed{UNSOLVABLE}", ref)[0]
    assert score_maze("Final:\n\\boxed{S220\\\\1121\\\\002E}", REF)[0]
