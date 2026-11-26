# r07-tokenmean-bridgemask — r06 최소 차이 대조군

시작 2026-11-26 18:45 UTC (`scripts/run_queue.sh`, W&B 태그 token-mean / bridge-masked / control-r06).
설정 `configs/tpu_v6e_7b_r07.yaml` = `tpu_v6e_7b.yaml` (r06) 에서 두 줄만 다름:

```
<   continuation: {enabled: true, min_tokens: 96, free_image_tokens: true, bridge: "..."}
>   continuation: {enabled: true, min_tokens: 96, free_image_tokens: true, mask_bridge: true, bridge: "..."}
>   loss_agg_mode: token-mean   # r06: Tunix 기본 sequence-mean-token-mean
```

1. `loss_agg_mode: token-mean` (GRPOConfig 에 전달).
2. `mask_bridge`: 강제 bridge 토큰 (18 토큰) 을 completion_mask 에서 뺀다 → policy gradient 없음. 이어쓰기 자체는 그대로.
   `tar_rlvr/train/continuation.py::install_bridge_mask`. 지표 `visual/bridge_masked_tokens`, `visual/bridge_continued` (step 합).
- 로그만 고침: `visual/reward_text` 는 텍스트 샘플이 없으면 NaN (r06 은 0).
- 2-step dry run: masked 162 = 18 × 9 (두 step), bridge 위치 토큰 검사 통과, LoRA 업데이트 ≠ 0. step 0 은 r06 과 같음 (img_frac 0.5625, reward 0.0625).

## step 40 (원시 step 값의 평균, 창당 n=20)

| metric | r06 0–19 | r06 20–39 | r07 0–19 | r07 20–39 |
|---|---|---|---|---|
| visual/img_frac | 0.625 | 0.906 | 0.447 | 0.447 |
| learning_signal/reward_mean | 0.219 | 0.297 | 0.222 | 0.300 |
| length/tokens_median | 394.3 | 383.8 | 424.9 | 332.4 |
| visual/bridge_masked_tokens (r07) | — | — | 126.0 | 126.0 |

- r07 의 두 창 img_frac 이 같은 건 우연 (각각 그림 응답 143/320). step 별 0.0–1.0.
- 매 step masked = 18 × continued.
