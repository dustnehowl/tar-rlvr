"""Tunix GRPO reward_fns. 시그니처: fn(prompts, completions, **cols) -> list[float]
데이터셋의 추가 컬럼(task, ref 등)이 **cols 로 넘어온다 (Tunix gsm8k 예제와 동일한 방식)."""
from __future__ import annotations
import json
from .parsing import extract_answer, has_answer_tag, count_visual_tokens, visual_blocks
from .verifiers import VERIFIERS


def _s(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "replace")
    x = str(x)
    if len(x) > 2 and x[0] == "b" and x[1] in "'\"" and x[-1] == x[1]:
        import ast
        try:
            return ast.literal_eval(x).decode("utf-8", "replace")
        except Exception:
            pass
    return x


def _refs(cols):
    return [json.loads(_s(r)) if not isinstance(r, dict) else r for r in cols["ref"]]


def correctness_reward(prompts, completions, **cols) -> list[float]:
    out = []
    for comp, task, ref in zip(completions, cols["task"], _refs(cols)):
        comp, task = _s(comp), _s(task)
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
        comp = _s(comp)
        if not has_answer_tag(comp):
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
            comp = _s(comp)
            blocks = visual_blocks(comp)
            # 답이 없으면 0: 그림만 그리고 끝내는 행동에 보상을 주면 그쪽으로 수렴한다 (파일럿에서 관찰됨)
            if not blocks or extract_answer(comp) is None:
                out.append(0.0); continue
            over = sum(max(0, count_visual_tokens(b) - max_tokens_per_image) for b in blocks)
            n_pen = max(0, len(blocks) - max_images)
            out.append(max(0.0, 1.0 - over / (max_tokens_per_image * max(1, len(blocks))) - 0.1 * n_pen))
        return out
    return visual_budget_reward


def _scaled(name, fn, w):
    def scaled(prompts, completions, **cols):
        return [w * r for r in fn(prompts, completions, **cols)]
    scaled.__name__ = name
    return scaled


def build_reward_fns(cfg: dict):
    fns = []
    if cfg.get("correctness", 0):
        fns.append(_scaled("correctness", correctness_reward, cfg["correctness"]))
    if cfg.get("format", 0):
        fns.append(_scaled("format", format_reward, cfg["format"]))
    if cfg.get("visual_budget", 0):
        vb = make_visual_budget_reward(cfg.get("max_visual_tokens_per_image", 128))
        fns.append(_scaled("visual_budget", vb, cfg["visual_budget"]))
    return fns