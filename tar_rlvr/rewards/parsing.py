"""Completion 파싱: <answer>…</answer> 추출, 시각 토큰 통계."""
from __future__ import annotations
import json, re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
VISUAL_TOKEN_RE = re.compile(r"<I(\d+)>")
IMAGE_BLOCK_RE = re.compile(r"<im_start>(.*?)<im_end>", re.S)  # Tar 스타일 이미지 블록 (필요시 수정)


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.findall(text)
    return m[-1].strip() if m else None


def parse_grid(s: str) -> list[list[int]] | None:
    """'[[0,1],[2,3]]' 또는 줄바꿈/공백 구분 그리드 → 2D int list."""
    s = s.strip()
    try:
        g = json.loads(s)
        if isinstance(g, list) and all(isinstance(r, list) for r in g):
            return [[int(x) for x in r] for r in g]
    except Exception:
        pass
    rows = [r for r in s.splitlines() if r.strip()]
    try:
        g = [[int(x) for x in re.findall(r"-?\d+", r)] for r in rows]
        return g if g and all(len(r) == len(g[0]) for r in g) else None
    except Exception:
        return None


def count_visual_tokens(text: str) -> int:
    return len(VISUAL_TOKEN_RE.findall(text))


def visual_blocks(text: str) -> list[str]:
    """이미지 블록 단위로 분리. 블록 마커가 없으면 연속된 시각토큰 run을 한 블록으로 간주."""
    blocks = IMAGE_BLOCK_RE.findall(text)
    if blocks:
        return blocks
    return re.findall(r"(?:<I\d+>)+", text)
