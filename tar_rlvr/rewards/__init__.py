"""Tunix GRPO reward_fns. 시그니처: fn(prompts, completions, **cols) -> list[float]
데이터셋의 추가 컬럼(task, ref 등)이 **cols 로 넘어온다 (Tunix gsm8k 예제와 동일한 방식)."""
from __future__ import annotations
import json
from .parsing import extract_answer, count_visual_tokens, visual_blocks
from .verifiers import VERIFIERS


def _refs(cols):
    refs = cols["ref"]
    return [json.loads(r) if isinstance(r, str) else r for r in refs]


def correctness_reward(prompts, completions, **cols) -> list[float]:
    out = []
    for comp, task, ref in zip(completions, cols["task"], _refs(cols)):
        ans = extract_answer(comp)
        if ans is None:
            out.append(0.0)
            continue
        out.append(VERIFIERS[task](ans, ref))
    return out


def format_reward(prompts, completions, **cols) -> list[float]:
    """<answer> 태그 정확히 1개 + (시각 토큰을 썼다면) 답 이전에 등장."""
    out = []
    for comp in completions:
        has_ans = comp.count("<answer>") == 1 and comp.count("</answer>") == 1
        if not has_ans:
            out.append(0.0); continue
        vis_before = count_visual_tokens(comp.split("<answer>")[0])
        vis_after = count_visual_tokens(comp.split("</answer>")[-1])
        if vis_after > 0:
            out.append(0.0)          # 답 뒤에 그림 = 형식 위반
        else:
            out.append(1.0 if vis_before > 0 else 0.5)  # 시각 추론 없이 답만 = 절반
    return out


def make_visual_budget_reward(max_tokens_per_image: int = 128, max_images: int = 8):
    """시각 토큰 남발 방지: 블록당 토큰 수 / 블록 수가 예산 안이면 1, 넘으면 선형 감점."""
    def visual_budget_reward(prompts, completions, **cols) -> list[float]:
        out = []
        for comp in completions:
            blocks = visual_blocks(comp)
            if not blocks:
                out.append(0.0); continue   # 시각 추론 안 쓰면 보너스 없음 (페널티도 없음)
            over = sum(max(0, count_visual_tokens(b) - max_tokens_per_image) for b in blocks)
            n_pen = max(0, len(blocks) - max_images)
            out.append(max(0.0, 1.0 - over / (max_tokens_per_image * max(1, len(blocks))) - 0.1 * n_pen))
        return out
    return visual_budget_reward


def build_reward_fns(cfg: dict):
    """config.rewards 의 가중치로 reward 리스트 구성. Tunix는 여러 fn 합산."""
    w = cfg
    fns = []
    if w.get("correctness", 0):
        fns.append(lambda p, c, **k: [w["correctness"] * r for r in correctness_reward(p, c, **k)])
    if w.get("format", 0):
        fns.append(lambda p, c, **k: [w["format"] * r for r in format_reward(p, c, **k)])
    if w.get("visual_budget", 0):
        vb = make_visual_budget_reward(w.get("max_visual_tokens_per_image", 128))
        fns.append(lambda p, c, **k: [w["visual_budget"] * r for r in vb(p, c, **k)])
    return fns
