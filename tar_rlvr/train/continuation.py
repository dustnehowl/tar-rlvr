"""2단계 롤아웃: 이미지 블록(<im_end>)으로 끝나고 답이 없는 completion 을 이어서 생성한다.

왜: Tar 는 이미지 생성 모델이라 <im_end> 직후 EOS 를 낸다. 그대로 두면 시각 샘플은 절대 답에 못 가서
correctness 기준으론 항상 0 → 시각 사용이 빠르게 억제된다. 그래서
  1) 1차 생성이 <im_end> 로 끝나고 답이 없으면
  2) prompt + 1차 결과 + bridge 를 프롬프트로 남은 토큰 예산만큼 2차 생성
  3) 학습 예제는 [1차 토큰(EOS 제거)] + [강제 bridge 토큰] + [2차 토큰] 으로 합친다.
bridge 는 "\\n" 만으로는 부족하다: Tar 는 <im_end> 뒤에 새 턴을 열고 또 그림을 그리는 사전분포가 강해서
(파일럿에서 8/8), "\\nBased on the sketch, " 처럼 텍스트 문장 중간으로 끌고 들어가는 접두어를 쓴다.
bridge 토큰이 completion 에 포함되므로 advantage>0 이면 P(bridge | …<im_end>) 가 올라가 EOS/재그림과 경쟁한다
→ 나중엔 강제 없이도 이미지 뒤에 이어 쓰는 행동이 발현될 수 있다. "그림 시작" 은 강제하지 않는다.

Tunix 의 RLCluster.generate 를 몽키패치한다 (logits/logprobs 는 GRPO 학습에 쓰이지 않음).
"""
from __future__ import annotations
import dataclasses
import numpy as np
import jax.numpy as jnp

from tar_rlvr.rewards.parsing import extract_answer

STATS = {"first_pass": 0, "continued": 0, "visual": 0, "visual_closed": 0, "redraw": 0}


def _n_real(tokens_row: np.ndarray, pad_id: int, eos_id: int) -> int:
    """EOS/pad 제외한 실제 토큰 수 (EOS 는 포함하지 않음)."""
    n = 0
    for t in tokens_row:
        if t == pad_id or t == eos_id:
            break
        n += 1
    return n


def needs_continuation(text: str, n_tokens: int, budget: int, min_cont: int) -> bool:
    if "<im_start>" not in text:
        return False
    if "<im_end>" not in text:                 # 이미지가 잘림 (S0 등) → 이어도 의미 없음
        return False
    if extract_answer(text) is not None and "<answer>" in text:
        return False
    return budget - n_tokens >= min_cont


def install_continuation(cluster, tok, max_completion: int, min_cont: int = 96,
                         bridge: str = "\nBased on the sketch, ", debug: bool = False):
    from tunix.rl import rl_cluster as rl_cluster_lib
    orig_generate = cluster.generate
    pad_id, eos_id = cluster.rollout.pad_id(), cluster.rollout.eos_id()
    bridge_ids = tok.encode(bridge, add_special_tokens=False) if hasattr(tok, "encode") else [ord(bridge) + 3]
    base_cfg = cluster.cluster_config.rollout_config

    def generate(prompts, apply_chat_template=False, mode=rl_cluster_lib.Mode.TRAIN, micro_batch_size=None):
        out = orig_generate(prompts, apply_chat_template=apply_chat_template, mode=mode, micro_batch_size=micro_batch_size)
        if mode != rl_cluster_lib.Mode.TRAIN:
            return out
        tokens = np.asarray(out.tokens)
        texts = list(out.text)
        n_real = [_n_real(tokens[i], pad_id, eos_id) for i in range(len(texts))]
        STATS["first_pass"] += len(texts)
        STATS["visual"] += sum("<im_start>" in t for t in texts)
        STATS["visual_closed"] += sum("<im_end>" in t for t in texts)
        idx = [i for i, t in enumerate(texts) if needs_continuation(t, n_real[i], max_completion, min_cont)]
        if not idx:
            return out
        STATS["continued"] += len(idx)
        remaining = max_completion - max(n_real[i] for i in idx) - len(bridge_ids)
        cfg2 = dataclasses.replace(base_cfg, max_tokens_to_generate=remaining,
                                   max_prompt_length=base_cfg.max_prompt_length + 256)
        prompts2 = [prompts[i] + texts[i] + bridge for i in idx]
        with cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ROLLOUT]:
            out2 = cluster.rollout.generate(prompts2, cfg2)
        tokens2 = np.asarray(out2.tokens)
        merged, merged_text = [], list(texts)
        for j, i in enumerate(idx):
            m = _n_real(tokens2[j], pad_id, eos_id)
            seq = list(tokens[i][: n_real[i]]) + bridge_ids + list(tokens2[j][: m])
            if m < tokens2.shape[1] and tokens2[j][m] == eos_id:
                seq.append(eos_id)
            merged.append((i, seq))
            merged_text[i] = texts[i] + bridge + out2.text[j]
            if "<im_start>" in out2.text[j]:
                STATS["redraw"] += 1   # 이어쓰기에서 또 그림을 그린 경우 (지표)
        L = max(tokens.shape[1], max(len(s) for _, s in merged))
        new = np.full((tokens.shape[0], L), pad_id, dtype=tokens.dtype)
        new[:, : tokens.shape[1]] = tokens
        for i, seq in merged:
            new[i, :] = pad_id
            new[i, : len(seq)] = np.asarray(seq, dtype=tokens.dtype)
        if debug:
            n_re = sum("<im_start>" in out2.text[j] for j in range(len(idx)))
            print(f"[continuation] {len(idx)}/{len(texts)} continued after image (budget {remaining}); redrew instead of reasoning: {n_re}")
        return dataclasses.replace(out, tokens=jnp.asarray(new), text=merged_text, logits=None, logprobs=None)

    cluster.generate = generate
    return cluster


def visual_metrics(prompts, completions, rewards, advantages=None, advances=None, **cols):
    """항상 켜두는 metric_fn: 시각 사용률 / 시각+답 도달률 / 시각 샘플 정답률 / 재그림률 — 연구의 핵심 지표."""
    comps = [c.decode("utf-8", "replace") if isinstance(c, (bytes, bytearray)) else str(c) for c in completions]
    rewards = np.asarray(rewards, dtype=float)
    vis = np.array(["<im_start>" in c for c in comps])
    ans = np.array([extract_answer(c) is not None for c in comps])
    redraw = np.array([c.count("<im_start>") >= 2 for c in comps])
    return {
        "visual/use_rate": (float(vis.mean()), np.mean),
        "visual/with_answer_rate": (float((vis & ans).sum() / max(1, vis.sum())), np.mean),
        "visual/reward_visual": (float(rewards[vis].mean()) if vis.any() else 0.0, np.mean),
        "visual/reward_text": (float(rewards[~vis].mean()) if (~vis).any() else 0.0, np.mean),
        "visual/redraw_rate": (float(redraw.sum() / max(1, vis.sum())), np.mean),
    }