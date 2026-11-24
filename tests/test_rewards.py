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

# --- 거짓 음성 점검 (boxed 중첩, 스레드 안 채점) ---
from tar_rlvr.rewards.parsing import boxed_contents

LONG = "Therefore, after simplifying everything carefully, the value we want is "


def _mc(letter):
    return {"answer": letter, "options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "choice": letter}


def _score(comp, ref):
    return correctness_reward(None, [comp], task=["math"], ref=[json.dumps(ref)])[0]


def test_boxed_nested_any_depth():
    assert boxed_contents(r"\boxed{\frac{\sqrt{3}}{2}}") == [r"\frac{\sqrt{3}}{2}"]
    assert boxed_contents(r"\boxed{\frac{\sqrt{\frac{1}{2}}}{2}}") == [r"\frac{\sqrt{\frac{1}{2}}}{2}"]
    assert boxed_contents(r"\boxed{1} then \fbox{x^{2}}") == ["1", "x^{2}"]
    assert boxed_contents(r"\boxed{\{1,2\}}") == [r"\{1,2\}"]
    assert boxed_contents(r"\boxed{\boxed{5}}") == [r"\boxed{5}"]
    assert boxed_contents(r"\boxed{unclosed") == []
    # 긴 줄 / 앞쪽에 다른 boxed 가 있어도 마지막 boxed 를 뽑아야 한다
    assert extract_answer(LONG + r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"
    assert extract_answer(r"First \boxed{1}. " + LONG + r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"


def test_requested_false_negative_cases():
    assert _score(r"so \boxed{A}\end{enumerate}", _mc("A")) == 1.0
    assert _score(r"so \boxed{B}.", _mc("B")) == 1.0
    assert _score(LONG + r"\boxed{64\sqrt{3} \text{ cm}^2}", {"answer": r"64\sqrt{3}"}) == 1.0
    assert _score(LONG + r"\boxed{\frac{\sqrt{3}}{2}}", {"answer": r"\frac{\sqrt{3}}{2}"}) == 1.0
    assert _score(LONG + r"\boxed{\frac{\sqrt{3}}{2}}", {"answer": r"\frac{\sqrt{2}}{2}"}) == 0.0


def test_scoring_inside_worker_thread():
    """Tunix 는 reward fn 을 메인이 아닌 스레드에서 부른다. math-verify 가 거기서도 동작해야 한다."""
    import threading
    cases = [(LONG + r"\boxed{\frac{\sqrt{3}}{2}}", {"answer": r"\frac{\sqrt{3}}{2}"}, 1.0),
             (r"Final Answer: $\boxed{2 \sqrt{3}}$ cm²", {"answer": r"2\sqrt{3}"}, 1.0),
             (r"Hence, the answer is **C. 24**.", {"answer": "C", "options": {"A": "20", "B": "25", "C": "24", "D": "30"}, "choice": "C"}, 1.0)]
    out = {}
    t = threading.Thread(target=lambda: out.update(r=[_score(c, r) for c, r, _ in cases]))
    t.start(); t.join()
    assert out["r"] == [want for *_, want in cases]


# --- 거짓 양성 점검 (tpu_run3 감사에서 나온 실제 사례) ---
def _mm(pred, ans):
    return math_match(pred, {"answer": ans})


def test_false_positives_rejected():
    assert _mm("y=2x-5", "2") == 0.0                 # math-verify 가 마지막 숫자만 뽑던 문제
    assert _mm("1:3", "1") == 0.0
    assert _mm("x < 5000", "5000") == 0.0            # 부등식 vs 값
    assert _mm("x > -k + 2", "2") == 0.0
    assert _mm("y + 1", "1") == 0.0
    assert _mm("1 / sqrt(2)", "1") == 0.0
    assert _mm("800 * tan(β)", "800") == 0.0
    assert _mm("m = 4", "y = 8x - 4") == 0.0
    assert _mm("1998", "1996") == 0.0                # 정수끼리는 근사 비교 안 함
    assert _mm("\\frac{4}{27}", "\\frac{9}{64}") == 0.0   # 옛 절대오차 0.01 이 통과시키던 것
    assert _mm("\\infty", "7") == 0.0
    assert _mm("√8", "8") == 0.0
    assert _mm("(2, 9]", "(8, 9]") == 0.0            # 마지막 숫자 9 만 비교하던 것
    assert _mm("(-1, 1]", "(-1, 1)") == 0.0
    assert _mm("(2, 2013)", "(2, 2014)") == 0.0
    assert _mm("(-3,0)", "(-1,0) or (3,0)") == 0.0   # 여러 답: 개수·일대일 대응
    assert _mm("17", "7 or 17") == 0.0
    assert _mm("20, 24", "24") == 0.0
    assert _mm("AC = 9, BC = 12", "12") == 0.0
    assert _mm("\\frac{t}{T}", "1-\\left[1-\\dfrac{t}{T} \\right]^2") == 0.0   # math-verify 대소문자 병합
    mc = {"answer": "B", "options": {"A": "0.1588", "B": "0.1587", "C": "1", "D": "2"}, "choice": "B"}
    assert math_match("**A. 0.1588**", mc) == 0.0    # 글자를 골랐으면 글자로 채점


def test_true_positives_kept():
    assert _mm("5.66 cm", "4\\sqrt{2}") == 1.0       # 소수 근사
    assert _mm("10.40", "6\\sqrt{3}") == 1.0
    assert _mm("(4,3)", "(4, 3)") == 1.0
    assert _mm("(3,0), (-1,0)", "(-1,0) or (3,0)") == 1.0
    assert _mm("\\pm\\sqrt{3}", "-\\sqrt{3} or \\sqrt{3}") == 1.0
    assert _mm("2:1", "2") == 1.0
    assert _mm("BC = \\frac{3}{5} \\cdot AB = \\frac{3}{5} \\cdot 20 = 12", "12") == 1.0
    assert _mm("The maximum value of z is 2", "2") == 1.0
    assert _mm("5000 \\text{ kilograms}", "5000") == 1.0
    assert _mm("64\\sqrt{3} \\text{ cm}^2", "64\\sqrt{3}") == 1.0
    assert _mm("2√3 cm²", "2\\sqrt{3}") == 1.0
    assert _mm("4 * sqrt(2)", "4\\sqrt{2}") == 1.0
    assert _mm("\\frac{n(n-1)}{2}", "n(n-1)/2") == 1.0
    assert _mm("RHOMBUS", "rhombus") == 1.0
    assert _mm("45^{\\circ}", "Angle A = 45 degrees (angles B and C cannot be determined).") == 1.0
    assert _mm("78.5", "The area occupied by the fountain is 78.5 square meters") == 1.0
    mc = {"answer": "C", "options": {"A": "20", "B": "25", "C": "24", "D": "30"}, "choice": "C"}
    for pred in ["**C. 24**", "\\text{C}", "\\text{(C)}", "should be C. 24 square units", "24"]:
        assert math_match(pred, mc) == 1.0, pred


def test_verifier_breakdown_rule_vs_mathverify():
    from tar_rlvr.rewards.verifiers import verifier_breakdown
    assert verifier_breakdown("4", {"answer": "4"}) == (1.0, 1.0)
    rule, mv = verifier_breakdown("\\frac{1}{2}", {"answer": "0.5"})      # 규칙(근사) 도 math-verify 도 맞음
    assert rule == 1.0 and mv == 1.0
    rule, mv = verifier_breakdown("2\\sqrt{2}", {"answer": "\\sqrt{8}"})  # 기호 동치는 math-verify 만 안다
    assert rule == 0.0 and mv == 1.0
    assert verifier_breakdown("B", {"answer": "3", "options": {"A": "2", "B": "3"}, "choice": "B"}) == (1.0, None)


def test_token_stats():
    import numpy as np
    from tar_rlvr.train.telemetry import TokInfo, token_stats
    ti = TokInfo(pad=0, eos=1, vis_lo=100, vis_hi=200, img_lo=105, im_start=100)
    rows = [np.array([5, 6, 100, 102, 110, 111, 101, 7, 1, 0]),   # 그림 1개, EOS 로 끝
            np.array([5, 6, 7, 8])]                              # EOS 없이 예산 소진 → 잘림
    a, b = token_stats(rows, ti, budget=4)
    assert (a["n"], a["n_img"], a["n_vis"], a["n_text"], a["truncated"]) == (8, 1, 2, 3, False)
    assert abs(a["first_img"] - 2 / 8) < 1e-9
    assert b["truncated"] and b["n_img"] == 0


def test_forced_image_marks_last_k_of_each_group():
    import numpy as np
    from tar_rlvr.train import forced_image

    class FakeLearner:
        def _generate_and_compute_advantage(self, training_input, mode=None):
            return training_input["prompts"]

    L = FakeLearner()
    forced_image.install(L, num_generations=4, per_group=1)
    out = L._generate_and_compute_advantage({"prompts": np.array(["a"] * 4 + ["b"] * 4), "task": np.array(["math"] * 8)})
    assert out == ["a", "a", "a", "a<im_start><S2>", "b", "b", "b", "b<im_start><S2>"]
    assert [forced_image.is_forced(p) for p in out].count(True) == 2
