"""단계별 콘솔 로그. --debug 로 켠다.

Tunix 에 끼울 수 있는 훅은 두 군데:
  (a) reward_fn  : 롤아웃 직후, completion 과 reward 별 값
  (b) metric_fn  : reward 합산 + advantage 계산 직후 (prompts, completions, rewards, advantages, **cols)
policy 업데이트(loss/kl)는 Tunix 진행바(actor_train_loss, actor_train_kl)에 찍힌다.
"""
from __future__ import annotations
import re
import numpy as np

W = 70
_step = {"n": 0}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ufffd]")   # 제어문자 + 디코딩 실패 마커
_IMG_RUN = re.compile(r"(<S\d>)?((?:<I\d+>)+)")


def banner(title: str):
    print("\n" + "=" * W + f"\n  {title}\n" + "=" * W)


def _collapse(text: str) -> str:
    """<S2><I1><I2>…(84개) → [IMG S2 ×84]  — 구조는 보이고 길이는 짧게."""
    def rep(m):
        scale = m.group(1) or ""
        n = m.group(2).count("<I")
        return f"[IMG{(' ' + scale) if scale else ''} ×{n}]"
    return _IMG_RUN.sub(rep, text)


def _s(x):
    """bytes → str, 제어문자는 '·' 로 (터미널 보호), 시각 토큰 run 은 접는다."""
    x = x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
    return _collapse(_CTRL.sub("·", x))


def _show(label, text, limit):
    text = _s(text)
    if len(text) > limit:
        text = text[: limit // 2] + f" …[{len(text) - limit}자 생략]… " + text[-limit // 2:]
    print(f"--- {label}\n{text}")


def make_logging_reward(fn, max_chars: int = 1200):
    """reward fn 을 감싸서 [2. 롤아웃] 을 찍는다. 첫 번째 reward fn 에만 감쌀 것 (롤아웃 중복 출력 방지)."""
    def wrapped(prompts, completions, **cols):
        _step["n"] += 1
        banner(f"[2. 롤아웃] step {_step['n']}  — {len(completions)}개 completion")
        _show("prompt (공통)", prompts[0], max_chars)
        for i, c in enumerate(completions):
            print(f"\n#{i}")
            _show("completion", c, max_chars)
        return fn(prompts, completions, **cols)
    wrapped.__name__ = fn.__name__
    return wrapped


def make_logging_metric(reward_names: list[str], num_generations: int):
    """metric_fn: [3. reward 합산] [4. advantage] 를 문제(그룹) 단위로 찍는다."""
    def metric_fn(prompts, completions, rewards, advantages=None, advances=None, **cols):
        advantages = advantages if advantages is not None else advances   # Tunix 0.1.5 키워드 오타 대응
        rewards = np.asarray(rewards); advantages = np.asarray(advantages)
        banner(f"[3→4. reward 합산 → advantage]  G={num_generations}")
        G = num_generations
        for gi in range(0, len(completions), G):
            print(f"\n문제 #{gi // G}")
            for j in range(gi, min(gi + G, len(completions))):
                c = _s(completions[j]).replace("\n", " ")
                print(f"  샘플 {j - gi}: reward={float(rewards[j]):+.3f}  adv={float(advantages[j]):+.3f}  …{c[-100:]}")
        print(f"\n배치 평균 reward={rewards.mean():.3f}")
        return {"debug/mean_reward": (float(rewards.mean()), np.mean)}
    return metric_fn