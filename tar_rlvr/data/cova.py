"""CoVA 의 수학 config 두 개 (math_vr, mathnet) 로더.

스키마 (train/test 공통): question, answer, interleaved_reasoning, images, metadata(JSON 문자열)
  - RL 에는 question + answer 만 쓴다 (interleaved_reasoning/images 는 SFT 용)
  - 객관식은 보기 글자로 채점하도록 ref 에 options/choice 를 넣는다
  - passrate_file 이 있으면 기본 모델이 가끔만 맞추는 문제로 거른다 (scripts/scan_passrate.py)
"""
from __future__ import annotations
import hashlib, json, re
from .synthetic import SYSTEM

MATH_CONFIGS = ["math_vr", "mathnet"]
_MULTIPART = re.compile(r"(^|\n)\s*\(?\d+\)?\s*[:：]")        # "1: 40\n2: 7 or 17" 같은 묶음 문제 답
# "- A. …" / "**A.** …" / "- **A**. …" / "A. …"  → (글자, 본문)
_OPTION = re.compile(r"^\s*-?\s*\*{0,2}([A-H])[.)]?\*{0,2}[.)]?\s+(.+?)\s*$", re.M)
# 질문 본문에 (I)(II) / (Ⅰ)(Ⅱ) / (1)(2) 소문제가 둘 이상 → 묶음 문제
_Q_PART = re.compile(r"\(\s*(?:I{1,3}|IV|[ⅠⅡⅢⅣ]|[1-4])\s*\)")
# 증명/서술형: verifier 로 채점 불가
_PROOF_Q = re.compile(r"\b(prove|show that|justif|describe all|explain why|may it happen|is it possible|"
                      r"does there exist|determine whether|verify that)\b", re.I)
_INEQ_ANS = re.compile(r"[<>≤≥]|\\le\b|\\ge\b|\\leq|\\geq")


def _parse_options(question: str) -> dict[str, str] | None:
    key = "Options:"
    body = question.split(key, 1)[1] if key in question else question   # 헤더 없는 "A. -2\nB. -1" 도 허용
    opts = {}
    for letter, text in _OPTION.findall(body):
        if letter not in opts:
            opts[letter] = text.strip().rstrip("*").strip()
    return opts if len(opts) >= 2 and "A" in opts else None


def _ref(row: dict) -> dict:
    ref = {"answer": str(row["answer"]).strip()}
    try:
        meta = json.loads(row.get("metadata") or "{}")
    except Exception:
        meta = {}
    answers = meta.get("answers")
    if isinstance(answers, list) and len(answers) > 1:
        ref["answers"] = [str(a).strip() for a in answers]
    opts = _parse_options(row["question"])
    if opts:
        ref["options"] = opts
        from tar_rlvr.rewards.verifiers import option_letter_for
        ref["choice"] = option_letter_for(ref["answer"], opts)
    return ref


def _keep(ref: dict, max_answer_chars: int, question: str = "") -> str | None:
    """제외 사유를 돌려준다 (None 이면 keep). verifier 가 채점 못 하는 것들을 거른다."""
    a = ref["answer"]
    if not a:
        return "empty"
    if len(a) > max_answer_chars:
        return "long"          # 증명형/서술형
    if "\n" in a or _MULTIPART.search(a) or len(set(_Q_PART.findall(question))) >= 2:
        return "multipart"     # (1)(2) / (I)(II) 묶음 문제
    if "options" in ref and not ref.get("choice"):
        return "mc_unmatched"  # 객관식인데 골드가 어느 보기인지 못 찾음
    if "options" not in ref and (_PROOF_Q.search(question) or _INEQ_ANS.search(a)):
        return "proof"         # 증명/서술형, "6 < S ≤ 10" 같은 부등식 답
    return None


def load_cova(cfg: dict) -> list[dict]:
    from datasets import load_dataset
    import collections
    configs = cfg.get("tasks") or MATH_CONFIGS
    max_chars = cfg.get("max_answer_chars", 60)
    rows = []
    for name in configs:
        ds = load_dataset(cfg["hf_id"], name, split=cfg.get("split", "train"))
        ds = ds.remove_columns([c for c in ("interleaved_reasoning", "images") if c in ds.column_names])
        why = collections.Counter()
        for r in ds:
            ref = _ref(r)
            reason = _keep(ref, max_chars, r["question"])
            if reason:
                why[reason] += 1
                continue
            rows.append({"prompts": f"{SYSTEM}\n{r['question']}", "task": "math", "ref": json.dumps(ref),
                         "qid": hashlib.md5(r["question"].encode()).hexdigest()[:12]})
            why["kept"] += 1
        print(f"[cova] {name}: {dict(why)} (of {len(ds)})")
    # pass-rate 스캔 결과로 난이도 필터 (scripts/scan_passrate.py 출력). 전부 맞거나 전부 틀리는 문제는 GRPO 신호가 0.
    if cfg.get("passrate_file"):
        pr = json.load(open(cfg["passrate_file"]))
        lo, hi = cfg.get("min_pass", 1), cfg.get("max_pass", None)
        before = len(rows)
        rows = [r for r in rows if r["qid"] in pr
                and lo <= pr[r["qid"]]["pass"] <= (hi if hi is not None else pr[r["qid"]]["k"] - 1)]
        print(f"[cova] passrate filter {cfg['passrate_file']}: {before} → {len(rows)} rows (pass in [{lo}, {hi or 'k-1'}])")
    # 참조 풀이가 너무 긴 문제 제외 (completion 예산 안에서 못 푸는 문제). scripts/token_stats.py 출력.
    if cfg.get("solution_len_file"):
        sl = json.load(open(cfg["solution_len_file"]))
        cap = cfg.get("max_solution_tokens", 768)
        before = len(rows)
        rows = [r for r in rows if sl.get(r["qid"], 0) <= cap]
        print(f"[cova] solution-length filter <= {cap}: {before} -> {len(rows)} rows")
    if cfg.get("shuffle", True):
        import random
        random.Random(cfg.get("seed", 0)).shuffle(rows)
    if cfg.get("n_samples"):
        rows = rows[: cfg["n_samples"]]
    return rows