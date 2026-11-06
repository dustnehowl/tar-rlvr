import json
from tar_rlvr.rewards import correctness_reward, format_reward, make_visual_budget_reward
from tar_rlvr.rewards.verifiers import math_match, option_letter_for
from tar_rlvr.rewards.parsing import extract_answer


def test_math_numeric_and_latex():
    assert math_match("60", {"answer": "60"}) == 1.0
    assert math_match("$\\frac{1}{2}$", {"answer": "0.5"}) == 1.0
    assert math_match("x = 3", {"answer": "3"}) == 1.0
    assert math_match("\\boxed{45}", {"answer": "45^\\circ"}) == 1.0
    assert math_match("61", {"answer": "60"}) == 0.0


def test_math_approx_and_latex_without_dollars():
    assert math_match("1.414", {"answer": "$\\sqrt{2}$"}) == 1.0
    assert math_match("1.414", {"answer": "\\sqrt{2}"}) == 1.0
    assert math_match("6.28", {"answer": "2\\pi"}) == 1.0
    assert math_match("9\\pi", {"answer": "9\\pi"}) == 1.0
    assert math_match("2", {"answer": "2\\pi"}) == 0.0     # $ 없는 LaTeX 를 2 로 오파싱하면 안 됨
    assert math_match("20\\pi", {"answer": "20"}) == 0.0


def test_math_multi_answer():
    ref = {"answer": "30 or 60"}
    assert math_match("30 or 60", ref) == 1.0
    assert math_match("60", ref) == 0.0
    ref2 = {"answer": "3", "answers": ["3", "-3"]}
    assert math_match("3, -3", ref2) == 1.0
    assert math_match("3", ref2) == 0.0


def test_multiple_choice():
    opts = {"A": "$(0,10)$", "B": "$(\\frac{1}{10},10)$", "C": "$(\\frac{1}{10},+\\infty)$",
            "D": "$(0, \\frac{1}{10}) \\cup (10, +\\infty)$"}
    gold = "\\therefore x \\in (0, \\frac{1}{10}) \\cup (10, +\\infty)"
    assert option_letter_for(gold, opts) == "D"
    assert option_letter_for("D", opts) == "D"
    ref = {"answer": gold, "options": opts, "choice": "D"}
    assert math_match("D", ref) == 1.0
    assert math_match("(D)", ref) == 1.0
    assert math_match("B", ref) == 0.0
    assert math_match("(0, \\frac{1}{10}) \\cup (10, +\\infty)", ref) == 1.0


def test_reward_fns_end_to_end():
    comps = [
        "<I1><I2><I3> let me sketch the triangle... <answer>60</answer>",
        "<answer>59</answer>",
        "no tags at all",
    ]
    cols = {"task": ["math"] * 3, "ref": [json.dumps({"answer": "60"})] * 3}
    assert correctness_reward(None, comps, **cols) == [1.0, 0.0, 0.0]
    assert format_reward(None, comps, **cols) == [1.0, 0.5, 0.0]
    vb = make_visual_budget_reward(max_tokens_per_image=2)
    r = vb(None, comps, **cols)
    assert 0.0 < r[0] < 1.0 and r[1] == 0.0


def test_extract_last_answer():
    assert extract_answer("<answer>a</answer> wait <answer>b</answer>") == "b"




def test_extract_answer_fallbacks():
    assert extract_answer("blah blah\nAnswer:C") == "C"
    assert extract_answer("So the answer is C.\nAnswer:C") == "C"
    assert extract_answer("Answer is \\frac{8}{3}.") == "\\frac{8}{3}"
    assert extract_answer("12.57") == "12.57"
    assert extract_answer("… gives us a minimum value of \\( (6 - 2\\sqrt{5})\\pi \\). So the answer is C.") == "C"
    assert extract_answer("thus $x = \\boxed{42}$ done") == "42"
    long = "a very long rambling sentence that just keeps going without concluding anything " * 3
    assert extract_answer(long) is None