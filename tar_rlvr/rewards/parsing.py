"""Completion 파싱: 답 추출(관대한 fallback 포함), 시각 토큰 통계."""
from __future__ import annotations
import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
ANSWER_LINE_RE = re.compile(
    r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\s*[:=]?\s*(.+?)\s*$", re.I | re.M
)
VISUAL_TOKEN_RE = re.compile(r"<I(\d+)>")
IMAGE_BLOCK_RE = re.compile(r"<im_start>(.*?)<im_end>", re.S)


def _strip(s: str) -> str:
    s = s.strip().rstrip(".。").strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()   # \( … \)
    return s.strip("$ ").strip()


def extract_answer(text: str) -> str | None:
    """우선순위: <answer> 태그 > \\boxed{} > 'Answer: X' 줄 > 마지막 줄이 짧으면 그것.
    태그 없이도 답을 뽑아야 RL 초기에 reward 신호가 생긴다 (태그 사용은 format reward 가 유도)."""
    m = ANSWER_RE.findall(text)
    if m:
        return _strip(m[-1])
    m = BOXED_RE.findall(text)
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