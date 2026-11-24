"""Completion 파싱: 답 추출(관대한 fallback 포함), 시각 토큰 통계."""
from __future__ import annotations
import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_BOXED_START_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
ANSWER_LINE_RE = re.compile(
    r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\s*[:=]?\s*(.+?)\s*$", re.I | re.M
)
VISUAL_TOKEN_RE = re.compile(r"<I(\d+)>")
IMAGE_BLOCK_RE = re.compile(r"<im_start>(.*?)<im_end>", re.S)


def _strip(s: str) -> str:
    s = s.strip().rstrip(".。").strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()   # \( … \)
    return s.strip("$ ").strip()


def boxed_contents(text: str) -> list[str]:
    """모든 \\boxed{…} / \\fbox{…} 의 내용을 등장 순서대로. 괄호 카운팅이라 중첩 깊이 제한 없음
    (정규식은 \\boxed{\\frac{\\sqrt{3}}{2}} 같은 2단계 중첩에서 실패했다). 이스케이프된 \\{ \\} 는 세지 않고,
    닫히지 않은 boxed 는 버린다."""
    out, covered = [], 0
    for m in _BOXED_START_RE.finditer(text):
        if m.start() < covered:    # 바깥 boxed 안의 boxed 는 따로 세지 않음
            continue
        depth, i, start = 1, m.end(), m.end()
        while i < len(text):
            c = text[i]
            if c == "\\":          # \{ \} \\ 등 이스케이프: 다음 글자 건너뜀
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:i])
                    covered = i
                    break
            i += 1
    return out


def extract_answer(text: str) -> str | None:
    """우선순위: <answer> 태그 > \\boxed{} > 'Answer: X' 줄 > 마지막 줄이 짧으면 그것.
    태그 없이도 답을 뽑아야 RL 초기에 reward 신호가 생긴다 (태그 사용은 format reward 가 유도)."""
    m = ANSWER_RE.findall(text)
    if m:
        return _strip(m[-1])
    m = boxed_contents(text)
    if m:
        return _strip(m[-1])
    m = ANSWER_LINE_RE.findall(text)
    if m:
        return _strip(m[-1])
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines and len(lines[-1]) <= 40:
        return _strip(lines[-1])
    return None


def has_answer_tag(text: str) -> bool:
    return text.count("<answer>") == 1 and text.count("</answer>") == 1


def count_visual_tokens(text: str) -> int:
    return len(VISUAL_TOKEN_RE.findall(text))


def visual_blocks(text: str) -> list[str]:
    blocks = IMAGE_BLOCK_RE.findall(text)
    if blocks:
        return blocks
    return re.findall(r"(?:<I\d+>)+", text)