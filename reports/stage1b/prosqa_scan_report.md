# Stage 1 — oracle on demand (logs/stage1b/prosqa)

프롬프트 100 (held-out 0), 객관식 0, leaky (휴리스틱) 0 → main = leaky 제외 100.


## sampled (8 샘플, T=0.8)

<im_start> 를 낸 프롬프트: 40/100 (샘플 기준 72/800)


### 조건별 pass@1 / pass@8 (%, 95% CI)

| subset | n | text_only | model_drawn |
|---|---|---|---|
| main (non-leaky) | 100 | 25.9 / 79.0 | 25.5 / 76.0 |
| main · free-form | 100 | 25.9 / 79.0 | 25.5 / 76.0 |
| main · emitted | 40 | 29.1 / 82.5 | 25.3 / 80.0 |
| main · emitted · free-form | 40 | 29.1 / 82.5 | 25.3 / 80.0 |
| main · not emitted | 60 | 23.8 / 76.7 | 25.6 / 73.3 |
| main · not emitted · free-form | 60 | 23.8 / 76.7 | 25.6 / 73.3 |
| all 300 | 100 | 25.9 / 79.0 | 25.5 / 76.0 |

### 짝 비교 (A − B)

| subset | n | A − B | Δpass@1 (pp) [95% CI] | Δpass@8 (pp) [95% CI] |
|---|---|---|---|---|
| main (non-leaky) | 100 | model_drawn − text_only | -0.4 [-4.5, +3.9] | -3.0 [-11.0, +6.0] |
| main · free-form | 100 | model_drawn − text_only | -0.4 [-4.5, +3.9] | -3.0 [-11.0, +6.0] |
| main · emitted | 40 | model_drawn − text_only | -3.8 [-9.4, +1.9] | -2.5 [-15.0, +10.0] |
| main · emitted · free-form | 40 | model_drawn − text_only | -3.8 [-9.4, +1.9] | -2.5 [-15.0, +10.0] |
| main · not emitted | 60 | model_drawn − text_only | +1.9 [-3.5, +7.5] | -3.3 [-15.0, +8.3] |
| main · not emitted · free-form | 60 | model_drawn − text_only | +1.9 [-3.5, +7.5] | -3.3 [-15.0, +8.3] |

### <im_start> 를 낸 샘플만 (샘플 단위, 그림 세 조건; prefix 공유)

| subset | n 샘플 | model_drawn |  |
|---|---|---|
| main (non-leaky) | 72 (40 프롬프트) | 11.1 |  |
| main · free-form | 72 (40 프롬프트) | 11.1 |  |

### <im_start> 위치 (72 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 34
- 분포: 0–0: 71, 1–5: 0, 6–20: 0, 21–50: 1, 51–100: 0, 101–200: 0, 201–768: 0
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 55.1 | 7.0 | 0.00 | 0.0 | 64 |
| model_drawn | 56.2 | 5.9 | 0.09 | 0.0 | 11 |
