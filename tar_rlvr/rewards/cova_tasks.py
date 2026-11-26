"""CoVA 비수학 태스크 (maze, clutrr, prosqa) 채점. Stage 1b 용.

수학과 달리 답 추출을 태스크마다 다르게 한다 (maze 답은 여러 줄 격자라 extract_answer 로는 못 뽑음).
진입점: score_completion(task, completion, ref) -> (correct, has_answer, pred)

ref 형식
  maze  : {"answer": gold (격자 또는 "UNSOLVABLE"), "raw_maze": "S000\\n1100\\n...E"}
  clutrr: {"answer": "grandmother"}            (CoVA 표기 그대로, "neice" 오타 포함)
  prosqa: {"answer": "Tom is a worpus.", "options": ["worpus", "quimpus"], "gold": "worpus"}

채점 규칙
  maze  : gold 가 UNSOLVABLE 이면 pred 도 UNSOLVABLE. 아니면 pred 격자가 유효한 경로여야 한다:
          크기 같음, 벽(1)은 그대로, 열린 칸(0)은 0 또는 2, S/E 는 그대로 또는 2,
          {2, S, E} 칸들이 S–E 를 4방향으로 잇고, 모든 2 칸이 S 에서 닿는다 (따로 떨어진 2 없음).
          gold 와 같은 경로일 필요는 없다 (최단 경로가 여러 개일 수 있음).
  clutrr: 답에서 관계어를 찾아 (긴 것부터: granddaughter > daughter) 마지막 것이 gold 와 같으면 정답.
          niece/neice 는 같은 것으로 본다. 방향(A 가 B 의 X 인지, B 가 A 의 X 인지)은 CoVA gold 를 그대로 따른다.
  prosqa: 두 보기 중 답에 나온 것. 둘 다 나오면 "not (a) X" 로 부정된 것을 빼고, 그래도 둘이면 오답.
"""
from __future__ import annotations
import re
from collections import deque
from .parsing import extract_answer

# ---------------------------------------------------------------- maze
_GRID_LINE = re.compile(r"^[\s`|]*([SE012](?:[\s,]*[SE012])*)[\s`|]*$")


def _grid_blocks(text: str):
    """연속된 격자 줄 묶음들 → [(end_pos, [row, ...])]. 칸 사이 공백/쉼표 허용, \\boxed 안의 \\\\ 줄바꿈 허용."""
    text = text.replace("\\\\", "\n").replace("\\n", "\n")
    text = re.sub(r"\\boxed\s*\{|\\(?:begin|end)\{[a-z]*\}|\\text\{|[{}$]", "\n", text)
    blocks, cur, end = [], [], 0
    pos = 0
    for line in text.split("\n"):
        pos += len(line) + 1
        m = _GRID_LINE.match(line)
        if m:
            cur.append(re.sub(r"[\s,]", "", m.group(1)))
            end = pos
        else:
            if len(cur) >= 2:
                blocks.append((end, cur))
            cur = []
    if len(cur) >= 2:
        blocks.append((end, cur))
    return blocks


def extract_maze(text: str, W: int, H: int):
    """마지막 격자 (H 줄 × W 칸) 또는 UNSOLVABLE 중 텍스트에서 더 뒤에 나온 것. 없으면 None."""
    text = text.split("<im_end>")[-1]          # 그림 토큰 뒤 텍스트만
    from .parsing import boxed_contents
    bx = boxed_contents(text)
    if bx:                                     # \boxed{} 가 있으면 그 안만 답으로 본다 (본문의 'UNSOLVABLE' 언급은 무시)
        b = bx[-1]
        if re.search(r"unsolvable", b, re.I):
            return "UNSOLVABLE"
        rows = [r for r in (re.sub(r"[\s,]", "", x) for x in b.replace("\\\\", "\n").split("\n")) if r]
        rows = [r for r in rows if re.fullmatch(r"[SE012]+", r)]
        if len(rows) == H and all(len(r) == W for r in rows):
            return "\n".join(rows)
        return "INVALID"                       # 상자 안 답이 격자도 UNSOLVABLE 도 아님 → 답은 있으나 오답
    best = None
    for end, rows in _grid_blocks(text):
        # 블록 안에서 끝쪽 H 줄이 W 칸이면 채택 (앞에 다른 줄이 붙은 경우 대비)
        rows = [r for r in rows if len(r) == W]
        if len(rows) >= H:
            best = (end, "\n".join(rows[-H:]))
    u = [m.end() for m in re.finditer(r"unsolvable", text, re.I)]
    if u and (best is None or u[-1] > best[0]):
        # 'not unsolvable' / 'is it unsolvable?' 같은 문장은 드물다 → 마지막 언급을 답으로 본다
        return "UNSOLVABLE"
    return best[1] if best else None


def maze_valid(pred: str, raw: str) -> bool:
    R = raw.strip().split("\n")
    P = pred.strip().split("\n")
    if len(P) != len(R) or any(len(p) != len(r) for p, r in zip(P, R)):
        return False
    H, W = len(R), len(R[0])
    on, s, e = set(), None, None
    for y in range(H):
        for x in range(W):
            r, p = R[y][x], P[y][x]
            if r == "S":
                s = (y, x)
            if r == "E":
                e = (y, x)
            if r == "1":
                if p != "1":
                    return False
            elif r == "0":
                if p not in "02":
                    return False
            elif p not in (r, "2"):
                return False
            if p in "2SE" and r != "1":
                on.add((y, x))
    if s is None or e is None:
        return False
    seen, q = {s}, deque([s])
    while q:
        y, x = q.popleft()
        for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if n in on and n not in seen:
                seen.add(n); q.append(n)
    return e in seen and seen == on


def maze_from_question(question: str) -> str:
    m = re.search(r"Maze:\n((?:[SE01]+\n?)+)", question)
    assert m, question[:200]
    return m.group(1).strip()


def score_maze(text: str, ref: dict):
    raw = ref["raw_maze"].strip().split("\n")
    pred = extract_maze(text, len(raw[0]), len(raw))
    if pred is None:
        return False, False, None
    if ref["answer"].strip() == "UNSOLVABLE":
        return pred == "UNSOLVABLE", True, pred
    return pred != "UNSOLVABLE" and maze_valid(pred, ref["raw_maze"]), True, pred


# ---------------------------------------------------------------- clutrr
_REL_SYNONYM = {"niece": "neice"}


def _rel_canon(w: str) -> str:
    w = w.lower().strip()
    return _REL_SYNONYM.get(w, w)


# CoVA clutrr train 의 gold 전부 (2,688 행) + 철자 변형
CLUTRR_RELATIONS = sorted({
    "aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law", "granddaughter", "grandfather",
    "grandmother", "grandson", "husband", "mother", "mother-in-law", "neice", "niece", "nephew", "sister",
    "son", "son-in-law", "uncle", "wife", "sister-in-law", "brother-in-law",
}, key=len, reverse=True)
_REL_RE = re.compile(r"(?<![a-z-])(" + "|".join(re.escape(r) for r in CLUTRR_RELATIONS) + r")s?(?![a-z-])", re.I)


def extract_clutrr(text: str):
    text = text.split("<im_end>")[-1]
    a = extract_answer(text)
    for src in ([a] if a else []) + ["\n".join(text.strip().splitlines()[-2:])]:
        m = _REL_RE.findall(src)
        if m:
            return _rel_canon(m[-1])
    return None


def score_clutrr(text: str, ref: dict):
    pred = extract_clutrr(text)
    if pred is None:
        return False, False, None
    return pred == _rel_canon(ref["answer"]), True, pred


# ---------------------------------------------------------------- prosqa
_PROSQA_Q = re.compile(r"Is (\w+) an? (\w+) or (\w+)\?\s*$")


def prosqa_ref(question: str, answer: str) -> dict:
    m = _PROSQA_Q.search(question.strip())
    g = re.search(r"is an? (\w+)\.?\s*$", answer.strip())
    assert m and g, (question[-80:], answer)
    assert g.group(1) in (m.group(2), m.group(3)), (m.groups(), answer)
    return {"answer": answer.strip(), "options": [m.group(2), m.group(3)], "gold": g.group(1), "entity": m.group(1)}


def extract_prosqa(text: str, options):
    text = text.split("<im_end>")[-1]
    a = extract_answer(text)
    for src in ([a] if a else []) + ["\n".join(text.strip().splitlines()[-2:])]:
        found = [o for o in options if re.search(rf"\b{re.escape(o)}\b", src, re.I)]
        if len(found) == 2:
            found = [o for o in found if not re.search(rf"\bnot\s+(?:an?\s+)?{re.escape(o)}\b", src, re.I)]
        if len(found) == 1:
            return found[0]
        if len(found) == 2:
            return "BOTH"
    return None


def score_prosqa(text: str, ref: dict):
    pred = extract_prosqa(text, ref["options"])
    if pred is None:
        return False, False, None
    return pred == ref["gold"], True, pred


SCORERS = {"maze": score_maze, "clutrr": score_clutrr, "prosqa": score_prosqa}


def score_completion(task: str, text: str, ref: dict):
    return SCORERS[task](text, ref)
