"""수학 답 verifier. 순수 파이썬 + math-verify → pytest 로 하드웨어 없이 검증.

verifier: (pred_answer: str, ref: dict) -> float in [0, 1]
ref["answer"]  : 골드 답 문자열 (예: "60", "$\\frac{1}{2}$", "30 or 60")
ref["answers"] : (선택) 여러 정답이 모두 필요한 경우 리스트 — Math-VR metadata.answers 에서 옴
ref["options"] : (선택) 객관식 보기 {"A": "...", ...}
ref["choice"]  : (선택) 골드에 해당하는 보기 글자
"""
from __future__ import annotations
import re

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER = re.compile(r"^\(?([A-H])\)?[.)]?$")
_LATEXY = re.compile(r"[\\^{}]")


def _clean(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(therefore|because)\s*", "", s)
    s = re.sub(r"^[a-zA-Z]\s*(\\in|=)\s*", "", s)   # "x \in (…)" / "x = 3" → "(…)" / "3"
    return s.strip()


def _latex(s: str) -> str:
    """math-verify 는 $…$ 가 없으면 LaTeX 로 안 본다 ("2\\pi" → 2 로 파싱!). LaTeX 같으면 감싼다."""
    return f"${s}$" if _LATEXY.search(s) else s


def _parse(s: str):
    from math_verify import parse
    return parse(_latex(_clean(s)))


def _split_gold(ref: dict) -> list[str]:
    if ref.get("answers"):
        return [str(a) for a in ref["answers"]]
    a = str(ref["answer"]).strip()
    parts = re.split(r"\s+or\s+|\s*;\s*", a)
    return [p.strip() for p in parts if p.strip()] or [a]


def _value(s: str) -> float:
    import sympy
    for x in _parse(s):
        if not isinstance(x, str):
            return float(sympy.N(x))
    return float(_NUM.findall(s)[-1])


def _numeric_close(a: str, b: str, rel: float = 1e-2) -> bool:
    """math-verify 가 못 잡는 sqrt(2) vs 1.414 같은 근사 비교."""
    try:
        va, vb = _value(a), _value(b)
        return abs(va - vb) <= rel * max(1.0, abs(va))
    except Exception:
        return False


def _equiv(pred: str, gold: str) -> bool:
    if _clean(pred) == _clean(gold):
        return True
    try:
        from math_verify import verify
        if verify(_parse(gold), _parse(pred)):
            return True
    except Exception:
        pass
    return _numeric_close(pred, gold)


def option_letter_for(gold: str, options: dict[str, str]) -> str | None:
    """골드 답이 어느 보기인지. 골드가 이미 글자면 그대로."""
    m = _LETTER.match(gold.strip())
    if m and m.group(1) in options:
        return m.group(1)
    g = _clean(gold)
    for letter, text in options.items():
        if _clean(text) == g:
            return letter
    for letter, text in options.items():
        if _equiv(text, gold):
            return letter
    return None


def math_match(pred: str, ref: dict) -> float:
    pred = pred.strip()
    if ref.get("options") and ref.get("choice"):
        m = _LETTER.match(pred)
        if m:
            return float(m.group(1) == ref["choice"])
        return float(_equiv(pred, ref["options"][ref["choice"]]))   # 글자 대신 보기 내용을 쓴 경우
    golds = _split_gold(ref)
    if len(golds) == 1:
        return float(_equiv(pred, golds[0]))
    pred_parts = [p.strip() for p in re.split(r"\s+or\s+|\s*[;,]\s*", pred) if p.strip()] or [pred]
    return float(all(any(_equiv(pp, g) for pp in pred_parts) for g in golds))


VERIFIERS = {"math": math_match}