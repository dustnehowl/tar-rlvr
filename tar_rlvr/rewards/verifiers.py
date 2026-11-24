"""수학 답 verifier. 순수 파이썬 + math-verify → pytest 로 하드웨어 없이 검증.

verifier: (pred_answer: str, ref: dict) -> float in [0, 1]
ref["answer"]  : 골드 답 문자열 (예: "60", "$\\frac{1}{2}$", "30 or 60")
ref["answers"] : (선택) 여러 정답이 모두 필요한 경우 리스트 — Math-VR metadata.answers 에서 옴
ref["options"] : (선택) 객관식 보기 {"A": "...", ...}
ref["choice"]  : (선택) 골드에 해당하는 보기 글자
"""
from __future__ import annotations
import contextvars, re

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


# math-verify 의 기본 timeout 은 signal.alarm 기반이라 메인 스레드에서만 동작한다.
# Tunix 는 reward fn 을 ThreadPoolExecutor 안에서 부르므로, 기본값이면 parse/verify 가 매번 예외 →
# 아래 except 에 묻혀 학습 내내 문자열 일치로만 채점됐다 (tpu_run3). timeout 을 끈다:
# 답은 cova 로더에서 max_answer_chars(60) 이하로 걸러져 sympy 가 멈출 위험은 낮다.
_MV_KW_PARSE = {"parsing_timeout": None}
_MV_KW_VERIFY = {"timeout_seconds": None}
_warned = set()
_NO_MV = contextvars.ContextVar("_NO_MV", default=False)   # verifier_breakdown 의 "규칙만" 채점용


def _warn_once(where: str, e: Exception):
    key = (where, type(e).__name__)
    if key not in _warned:
        _warned.add(key)
        print(f"[verifier] math-verify {where} 실패 ({type(e).__name__}: {e}) — 이후 같은 종류는 생략")


def _parse(s: str):
    from math_verify import parse
    return parse(_latex(_clean(s)), **_MV_KW_PARSE)


# ---------------------------------------------------------------------------
# 데이터 로딩용 (option_letter_for) 옛 비교. 관대하지만 골드↔보기 매칭에만 쓰고 채점엔 쓰지 않는다.
# 바꾸면 cova 필터 결과(어떤 문제가 남는지, choice 글자)가 달라져 run 간 비교가 깨지므로 그대로 둔다.
def _legacy_value(s: str) -> float:
    import sympy
    for x in _parse(s):
        if not isinstance(x, str):
            return float(sympy.N(x))
    return float(_NUM.findall(s)[-1])


def _legacy_equiv(pred: str, gold: str) -> bool:
    if _clean(pred) == _clean(gold):
        return True
    try:
        from math_verify import verify
        if verify(_parse(gold), _parse(pred), **_MV_KW_VERIFY):
            return True
    except Exception as e:
        _warn_once("verify", e)
    try:
        va, vb = _legacy_value(pred), _legacy_value(gold)
        return abs(va - vb) <= 1e-2 * max(1.0, abs(va))
    except Exception:
        return False


def legacy_math_match(pred: str, ref: dict) -> float:
    """수정 전(tpu_run3 까지) 채점 그대로 — scripts/rescore_log.py 가 옛 로그를 재현할 때만 쓴다."""
    pred = pred.strip()
    if ref.get("options") and ref.get("choice"):
        m = _LETTER.match(pred)
        if m:
            return float(m.group(1) == ref["choice"])
        return float(_legacy_equiv(pred, ref["options"][ref["choice"]]))
    if ref.get("answers"):
        golds = [str(a) for a in ref["answers"]]
    else:
        a = str(ref["answer"]).strip()
        golds = [x.strip() for x in re.split(r"\s+or\s+|\s*;\s*", a) if x.strip()] or [a]
    if len(golds) == 1:
        return float(_legacy_equiv(pred, golds[0]))
    parts = [x.strip() for x in re.split(r"\s+or\s+|\s*[;,]\s*", pred) if x.strip()] or [pred]
    return float(all(any(_legacy_equiv(pp, g) for pp in parts) for g in golds))


# ---------------------------------------------------------------------------
# 채점용 비교. tpu_run3 감사에서 드러난 거짓 양성을 막는다:
#  - math-verify 는 $…$ 없는 텍스트에서 마지막 숫자만 뽑는다 ("y=2x-5"→2, "(-1, 1]"→1) → 예측은 항상 LaTeX 로 파싱
#  - 옛 근사 비교는 문자열의 마지막 숫자 + 절대오차 0.01 ("(2,9]"≈"(8,9]", 4/27≈9/64) → 소수 근사일 때만, 상대오차
#  - 여러 답은 괄호 밖 구분자로만 나누고 개수가 같아야 한다 ("17" ≠ "7 or 17", "20, 24" ≠ "24")
_INEQ = re.compile(r"[<>≤≥≠]|\\(?:le|ge|leq|geq|neq|lt|gt)\b")
_DECIMAL = re.compile(r"\d\.\d")
_UNIT = (r"(?:square\s+|sq\.?\s*|cubic\s+)?(?:units?|cm|mm|km|m|meters?|metres?|kilometers?|kilometres?|"
         r"kilograms?|kg|grams?|g|seconds?|sec|s|minutes?|min|hours?|hrs?|h|degrees?|deg|yuan|dollars?|"
         r"liters?|litres?|l|ml|hectares?|ha|people|students?|times|pieces?|days?|years?|km/h|m/s)")
_TRAIL_UNIT = re.compile(rf"^(.*?[\d}}])\s*(?:\\(?:text|mathrm|,|;|\s)\s*)*\{{?\s*{_UNIT}\s*\}}?"
                         rf"\s*(?:\^\s*\{{?\s*[23]\s*\}}?|[²³])?\s*\.?$", re.I)
_WORD = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{3,}")


def _strip_text_cmds(s: str) -> str:
    return re.sub(r"\\[A-Za-z]+", " ", s)


def _norm(s: str) -> str:
    s = _clean(s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:left|right)\s*(?=[()\[\]|.]|\\[{}])", "", s)
    s = re.sub(r"\\[,;!: ]", " ", s)
    s = s.replace("−", "-").replace("×", "\\times ").replace("·", "\\cdot ").replace("π", "\\pi ")
    s = s.replace("°", "^\\circ").replace("²", "^2").replace("³", "^3").replace("∞", "\\infty ")
    s = s.replace("≤", "\\le ").replace("≥", "\\ge ")
    s = re.sub(r"(?:√|\bsqrt)\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[A-Za-z])", r"\\sqrt{\1}", s)
    s = re.sub(r"\^\{\\circ\}|\^\\circ|\\circ|\\degree", "", s) if re.search(r"\d\s*(?:\^\{?\\circ|\\circ|\\degree)", s) else s
    s = s.strip().rstrip(".").strip().strip("$").strip()
    m = _TRAIL_UNIT.match(s)
    if m and not _WORD.search(_strip_text_cmds(m.group(1))):
        s = m.group(1).strip()
    return s


def _canon(s: str) -> str:
    return re.sub(r"\s+", "", _norm(s)).replace("{", "").replace("}", "")


def _top_split(s: str) -> list[str]:
    """괄호 깊이 0 에서만 , ; 'or' 'and' 로 나눈다 — "(4, 3)" "(-\\infty, 2]" 는 한 덩어리."""
    parts, depth, cur, i = [], 0, "", 0
    while i < len(s):
        if s.startswith("\\{", i) or s.startswith("\\}", i):
            depth += 1 if s[i + 1] == "{" else -1
            cur += s[i:i + 2]; i += 2; continue
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth <= 0:
            m = re.match(r"\s*(?:,|;|\bor\b|\band\b|\\text\{\s*(?:or|and)\s*\})\s*", s[i:])
            if m and (c in ",;" or s[i].isspace() or s[i] == "\\"):
                parts.append(cur); cur = ""; i += m.end(); continue
        cur += c; i += 1
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _mv_parse(s: str, plain_fallback: bool):
    from math_verify import parse
    out = parse(f"${s}$", **_MV_KW_PARSE)
    if not out and plain_fallback:
        out = parse(s, **_MV_KW_PARSE)
    return out


def _num(parsed) -> float | None:
    import sympy
    for x in parsed:
        if isinstance(x, str):
            continue
        try:
            if getattr(x, "free_symbols", None):
                return None
            v = complex(sympy.N(x))
            if abs(v.imag) < 1e-12 and abs(v.real) != float("inf"):
                return v.real
        except Exception:
            return None
        return None
    return None


def _numeric_close(a: str, b: str, rel: float = 1e-2) -> bool:
    """sqrt(2) vs 1.414 같은 근사 비교. 한쪽이 소수 근사값이고, 양쪽 모두 자유변수 없는 단일 수일 때만, 상대오차."""
    if not (_DECIMAL.search(a) or _DECIMAL.search(b)):
        return False
    try:
        va, vb = _num(_mv_parse(a, False)), _num(_mv_parse(b, True))
    except Exception as e:
        _warn_once("parse", e)
        return False
    if va is None or vb is None:
        return False
    return abs(va - vb) <= rel * max(abs(vb), 1e-9)


def _ratio(s: str) -> str:
    m = re.fullmatch(r"\s*([^:=<>]+?)\s*:\s*([^:=<>]+?)\s*", s)
    return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" if m else s


def _equiv_one(p: str, g: str) -> bool:
    """단일 답 비교 (이미 _norm 된 문자열)."""
    if _canon(p) == _canon(g):
        return True
    if bool(_INEQ.search(p)) != bool(_INEQ.search(g)):
        return False
    if (":" in p) or (":" in g):
        p, g = _ratio(p), _ratio(g)
    p, g = _case_safe(p, g)
    if _NO_MV.get():
        return _numeric_close(p, g)
    try:
        from math_verify import verify
        gp, pp = _mv_parse(g, True), _mv_parse(p, False)
        if gp and pp and verify(gp, pp, **_MV_KW_VERIFY):
            return True
    except Exception as e:
        _warn_once("verify", e)
    return _numeric_close(p, g)


def _case_safe(p: str, g: str) -> tuple[str, str]:
    """math-verify 는 변수 대소문자를 합친다 (t/T → 1). 같은 글자가 대·소문자로 함께 나오면 대문자를 X_{0} 로 바꿔 구분."""
    letters = set(re.findall(r"(?<![\\A-Za-z])([A-Za-z])(?![A-Za-z])", p + " " + g))
    clash = {c for c in letters if c.isupper() and c.lower() in letters}
    if not clash:
        return p, g
    pat = re.compile(r"(?<![\\A-Za-z])([%s])(?![A-Za-z])" % "".join(sorted(clash)))
    return pat.sub(r"\1_{0}", p), pat.sub(r"\1_{0}", g)


def _gold_core(g: str) -> str:
    """골드가 문장이면 ("Angle A = 45 degrees (angles B …)", "… is 78.5 square meters") 핵심 값만."""
    if not _WORD.search(_strip_text_cmds(g)):
        return g
    core = re.split(r"(?<![<>!\\])=|\b(?:is|are|equals)\b", g, maxsplit=1)[-1] if re.search(r"=|\b(?:is|are|equals)\b", g) else g
    core = re.sub(r"\([^()]*[A-Za-z]{3,}[^()]*\)", "", core)      # 설명 괄호 제거
    core = _norm(core)
    return core if core and not _WORD.search(_strip_text_cmds(core)) else g


def _focus(p: str, g: str) -> str:
    """예측이 "BC = 3/5·AB = 12" / "the maximum value is 2" 처럼 말·등식이면 마지막 우변만 비교 대상으로."""
    if "=" in p and "=" not in g:
        p = re.split(r"(?<![<>!\\])=", p)[-1].strip()
    if _WORD.search(_strip_text_cmds(p)) and not _WORD.search(_strip_text_cmds(g)):
        tail = re.split(r"\b(?:is|are|equals|be)\b", p)[-1].strip()
        p = _norm(tail)
    return p


def _expand_pm(parts: list[str]) -> list[str]:
    out = []
    for x in parts:
        if "\\pm" in x:
            out += [x.replace("\\pm", "+", 1).lstrip("+"), x.replace("\\pm", "-", 1)]
        else:
            out.append(x)
    return out


def _equiv(pred: str, gold: str) -> bool:
    """채점용. gold 가 여러 답("30 or 60", "(3,0) or (-1,0)")이면 예측도 같은 개수여야 하고 일대일로 맞아야 한다."""
    p, g = _norm(pred), _norm(gold)
    if _canon(p) == _canon(g):
        return True
    gs = _top_split(g)
    if len(gs) <= 1:
        gs = [g]
    if len(gs) == 1:
        if len(_expand_pm(_top_split(p))) > 1:
            return False                      # "20, 24" / "AC = 9, BC = 12" / "\pm 3" vs 단일 골드
        g = _gold_core(g)
        if _canon(p) == _canon(g):
            return True
        return _equiv_one(_norm(_focus(p, g)), g)
    ps = _expand_pm(_top_split(p))
    if len(ps) != len(gs):
        return False
    used = set()
    for gg in gs:
        hit = next((k for k, pp in enumerate(ps) if k not in used and _equiv_one(_norm(pp), _norm(gg))), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def option_letter_for(gold: str, options: dict[str, str]) -> str | None:
    """골드 답이 어느 보기인지. 골드가 이미 글자면 그대로. (데이터 로딩용 — _legacy_equiv 유지)"""
    m = _LETTER.match(gold.strip())
    if m and m.group(1) in options:
        return m.group(1)
    g = _clean(gold)
    for letter, text in options.items():
        if _clean(text) == g:
            return letter
    for letter, text in options.items():
        if _legacy_equiv(text, gold):
            return letter
    return None


# "**C. 24**", "B. 0.16", "should be A. 6 square units", "\text{C}. 6" 처럼 글자를 고른 답 → 글자로 채점
_LETTER_LEAD = re.compile(r"^[\s*_(\[]*([A-H])\s*[).:]\s*(?:\*\*)?\s*\S")
_LETTER_ANY = re.compile(r"(?<![A-Za-z\\])\(?([A-H])\)?\.\s|\\text(?:bf)?\{\s*\(?\s*([A-H])\s*\)?\s*[.)]?\s*\}")


def _picked_letter(pred: str) -> str | None:
    m = _LETTER.match(pred.strip().strip("*").strip())
    if m:
        return m.group(1)
    m = _LETTER_LEAD.match(pred)
    if m:
        return m.group(1)
    hits = _LETTER_ANY.findall(pred)
    if hits:
        a, b = hits[-1]
        return a or b
    return None


def math_match(pred: str, ref: dict) -> float:
    pred = pred.strip()
    if ref.get("options") and ref.get("choice"):
        letter = _picked_letter(pred)
        if letter:
            return float(letter == ref["choice"])
        return float(_equiv(pred, ref["options"][ref["choice"]]))   # 글자 대신 보기 내용을 쓴 경우
    gold = " ; ".join(str(a) for a in ref["answers"]) if ref.get("answers") else str(ref["answer"])
    return float(_equiv(pred, gold))


def verifier_breakdown(pred: str, ref: dict) -> tuple[float, float | None]:
    """진단용 (학습 보상에는 안 씀): (규칙만 = math_match 에서 math-verify 를 뺀 판정, math-verify 만 = 골드 전체 vs 예측 전체).
    객관식에서 글자를 고른 답은 math-verify 비교가 무의미하므로 None."""
    tok = _NO_MV.set(True)
    try:
        rule = math_match(pred, ref)
    finally:
        _NO_MV.reset(tok)
    pred = pred.strip()
    if ref.get("options") and ref.get("choice"):
        if _picked_letter(pred):
            return rule, None
        gold = ref["options"][ref["choice"]]
    else:
        gold = " ; ".join(str(a) for a in ref["answers"]) if ref.get("answers") else str(ref["answer"])
    try:
        from math_verify import verify
        gp, pp = _mv_parse(_norm(gold), True), _mv_parse(_norm(pred), False)
        mv = float(bool(gp and pp and verify(gp, pp, **_MV_KW_VERIFY)))
    except Exception as e:
        _warn_once("verify", e)
        mv = 0.0
    return rule, mv


VERIFIERS = {"math": math_match}
