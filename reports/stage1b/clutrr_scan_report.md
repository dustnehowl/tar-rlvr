# Stage 1 — oracle on demand (logs/stage1b/clutrr)

프롬프트 200 (held-out 0), 객관식 0, leaky (휴리스틱) 7 → main = leaky 제외 193.


## sampled (8 샘플, T=0.8)

<im_start> 를 낸 프롬프트: 175/200 (샘플 기준 847/1600)


### 조건별 pass@1 / pass@8 (%, 95% CI)

| subset | n | text_only | model_drawn |
|---|---|---|---|
| main (non-leaky) | 193 | 11.3 / 45.1 | 10.6 / 43.5 |
| main · free-form | 193 | 11.3 / 45.1 | 10.6 / 43.5 |
| main · emitted | 169 | 11.3 / 45.6 | 10.5 / 42.0 |
| main · emitted · free-form | 169 | 11.3 / 45.6 | 10.5 / 42.0 |
| main · not emitted | 24 | 10.9 / 41.7 | 11.5 / 54.2 |
| main · not emitted · free-form | 24 | 10.9 / 41.7 | 11.5 / 54.2 |
| leaky only | 7 | 16.1 / 42.9 | 12.5 / 57.1 |
| all 300 | 200 | 11.4 / 45.0 | 10.7 / 44.0 |

### 짝 비교 (A − B)

| subset | n | A − B | Δpass@1 (pp) [95% CI] | Δpass@8 (pp) [95% CI] |
|---|---|---|---|---|
| main (non-leaky) | 193 | model_drawn − text_only | -0.6 [-2.7, +1.5] | -1.6 [-9.8, +6.7] |
| main · free-form | 193 | model_drawn − text_only | -0.6 [-2.7, +1.5] | -1.6 [-9.8, +6.7] |
| main · emitted | 169 | model_drawn − text_only | -0.8 [-3.0, +1.3] | -3.6 [-12.4, +4.7] |
| main · emitted · free-form | 169 | model_drawn − text_only | -0.8 [-3.0, +1.3] | -3.6 [-12.4, +4.7] |
| main · not emitted | 24 | model_drawn − text_only | +0.5 [-7.3, +8.3] | +12.5 [-12.5, +37.5] |
| main · not emitted · free-form | 24 | model_drawn − text_only | +0.5 [-7.3, +8.3] | +12.5 [-12.5, +37.5] |

### <im_start> 를 낸 샘플만 (샘플 단위, 그림 세 조건; prefix 공유)

| subset | n 샘플 | model_drawn |  |
|---|---|---|
| main (non-leaky) | 815 (169 프롬프트) | 10.7 |  |
| main · free-form | 815 (169 프롬프트) | 10.7 |  |

### <im_start> 위치 (847 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 67
- 분포: 0–0: 824, 1–5: 11, 6–20: 5, 21–50: 4, 51–100: 3, 101–200: 0, 201–768: 0
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 74.1 | 2.5 | 0.00 | 0.0 | 54 |
| model_drawn | 67.6 | 3.6 | 0.49 | 0.0 | 194 |
