# Stage 1 — oracle on demand (logs/stage1/base)

프롬프트 298 (held-out 100), 객관식 162, leaky (휴리스틱) 33 → main = leaky 제외 265.


## greedy (1 샘플)

<im_start> 를 낸 프롬프트: 184/298


### 조건별 greedy acc (%, 95% CI)

| subset | n | text_only | model_drawn | oracle | corrupt |
|---|---|---|---|---|---|
| main (non-leaky) | 265 | 27.2 [22.2, 32.8] | 20.4 [16.0, 25.6] | 23.0 [18.4, 28.5] | 20.4 [16.0, 25.6] |
| main · free-form | 115 | 23.5 [16.7, 32.0] | 20.9 [14.4, 29.2] | 24.3 [17.4, 32.9] | 26.1 [18.9, 34.8] |
| main · MC | 150 | 30.0 [23.2, 37.8] | 20.0 [14.4, 27.1] | 22.0 [16.1, 29.3] | 16.0 [11.0, 22.7] |
| main · emitted | 166 | 30.7 [24.2, 38.1] | 22.9 [17.2, 29.9] | 27.1 [20.9, 34.3] | 22.9 [17.2, 29.9] |
| main · emitted · free-form | 89 | 25.8 [17.9, 35.8] | 21.3 [14.1, 31.0] | 25.8 [17.9, 35.8] | 28.1 [19.8, 38.2] |
| main · emitted · MC | 77 | 36.4 [26.5, 47.5] | 24.7 [16.4, 35.4] | 28.6 [19.7, 39.5] | 16.9 [10.1, 26.8] |
| main · not emitted | 99 | 21.2 [14.3, 30.3] | 16.2 [10.2, 24.7] | 16.2 [10.2, 24.7] | 16.2 [10.2, 24.7] |
| main · not emitted · free-form | 26 | 15.4 [6.1, 33.5] | 19.2 [8.5, 37.9] | 19.2 [8.5, 37.9] | 19.2 [8.5, 37.9] |
| leaky only | 33 | 33.3 [19.8, 50.4] | 24.2 [12.8, 41.0] | 36.4 [22.2, 53.4] | 27.3 [15.1, 44.2] |
| all 300 | 298 | 27.9 [23.1, 33.2] | 20.8 [16.6, 25.8] | 24.5 [20.0, 29.7] | 21.1 [16.9, 26.1] |

### 짝 비교 (A − B)

| subset | n | A − B | diff (pp) | A만 맞음 / B만 맞음 | McNemar p |
|---|---|---|---|---|---|
| main (non-leaky) | 265 | oracle − text_only | -4.2 | 33 / 44 | 0.254 |
| main (non-leaky) | 265 | oracle − corrupt | +2.6 | 22 / 15 | 0.324 |
| main (non-leaky) | 265 | oracle − model_drawn | +2.6 | 26 / 19 | 0.371 |
| main (non-leaky) | 265 | model_drawn − text_only | -6.8 | 26 / 44 | 0.0414 |
| main (non-leaky) | 265 | corrupt − text_only | -6.8 | 26 / 44 | 0.0414 |
| main · free-form | 115 | oracle − text_only | +0.9 | 15 / 14 | 1 |
| main · free-form | 115 | oracle − corrupt | -1.7 | 7 / 9 | 0.804 |
| main · free-form | 115 | oracle − model_drawn | +3.5 | 13 / 9 | 0.523 |
| main · free-form | 115 | model_drawn − text_only | -2.6 | 12 / 15 | 0.701 |
| main · free-form | 115 | corrupt − text_only | +2.6 | 14 / 11 | 0.69 |
| main · MC | 150 | oracle − text_only | -8.0 | 18 / 30 | 0.111 |
| main · MC | 150 | oracle − corrupt | +6.0 | 15 / 6 | 0.0784 |
| main · MC | 150 | oracle − model_drawn | +2.0 | 13 / 10 | 0.678 |
| main · MC | 150 | model_drawn − text_only | -10.0 | 14 / 29 | 0.0315 |
| main · MC | 150 | corrupt − text_only | -14.0 | 12 / 33 | 0.00246 |
| main · emitted | 166 | oracle − text_only | -3.6 | 25 / 31 | 0.504 |
| main · emitted | 166 | oracle − corrupt | +4.2 | 22 / 15 | 0.324 |
| main · emitted | 166 | oracle − model_drawn | +4.2 | 26 / 19 | 0.371 |
| main · emitted | 166 | model_drawn − text_only | -7.8 | 18 / 31 | 0.0854 |
| main · emitted | 166 | corrupt − text_only | -7.8 | 18 / 31 | 0.0854 |
| main · emitted · free-form | 89 | oracle − text_only | +0.0 | 13 / 13 | 1 |
| main · emitted · free-form | 89 | oracle − corrupt | -2.2 | 7 / 9 | 0.804 |
| main · emitted · free-form | 89 | oracle − model_drawn | +4.5 | 13 / 9 | 0.523 |
| main · emitted · free-form | 89 | model_drawn − text_only | -4.5 | 10 / 14 | 0.541 |
| main · emitted · free-form | 89 | corrupt − text_only | +2.2 | 12 / 10 | 0.832 |
| main · emitted · MC | 77 | oracle − text_only | -7.8 | 12 / 18 | 0.362 |
| main · emitted · MC | 77 | oracle − corrupt | +11.7 | 15 / 6 | 0.0784 |
| main · emitted · MC | 77 | oracle − model_drawn | +3.9 | 13 / 10 | 0.678 |
| main · emitted · MC | 77 | model_drawn − text_only | -11.7 | 8 / 17 | 0.108 |
| main · emitted · MC | 77 | corrupt − text_only | -19.5 | 6 / 21 | 0.00592 |
| main · not emitted | 99 | oracle − text_only | -5.1 | 8 / 13 | 0.383 |
| main · not emitted | 99 | oracle − corrupt | +0.0 | 0 / 0 | 1 |
| main · not emitted | 99 | oracle − model_drawn | +0.0 | 0 / 0 | 1 |
| main · not emitted | 99 | model_drawn − text_only | -5.1 | 8 / 13 | 0.383 |
| main · not emitted | 99 | corrupt − text_only | -5.1 | 8 / 13 | 0.383 |
| main · not emitted · free-form | 26 | oracle − text_only | +3.8 | 2 / 1 | 1 |
| main · not emitted · free-form | 26 | oracle − corrupt | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | oracle − model_drawn | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | model_drawn − text_only | +3.8 | 2 / 1 | 1 |
| main · not emitted · free-form | 26 | corrupt − text_only | +3.8 | 2 / 1 | 1 |

### <im_start> 위치 (184 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 129
- 분포: 0–0: 180, 1–5: 0, 6–20: 0, 21–50: 2, 51–100: 1, 101–200: 1, 201–768: 0
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 94.0 | 13.6 | 0.00 | 0.0 | 339 |
| model_drawn | 94.0 | 11.7 | 0.61 | 0.0 | 346 |
| oracle | 94.0 | 14.3 | 0.63 | 0.0 | 380 |
| corrupt | 95.8 | 14.3 | 0.63 | 0.4 | 377 |

## sampled (8 샘플, T=0.8)

<im_start> 를 낸 프롬프트: 239/298 (샘플 기준 1200/2384)


### 조건별 pass@1 / pass@8 (%, 95% CI)

| subset | n | text_only | model_drawn | oracle | corrupt |
|---|---|---|---|---|---|
| main (non-leaky) | 265 | 24.1 / 72.8 | 17.9 / 66.4 | 21.7 / 69.8 | 22.7 / 72.8 |
| main · free-form | 115 | 21.8 / 62.6 | 15.2 / 54.8 | 19.2 / 57.4 | 21.7 / 61.7 |
| main · MC | 150 | 25.8 / 80.7 | 20.0 / 75.3 | 23.6 / 79.3 | 23.4 / 81.3 |
| main · emitted | 213 | 24.4 / 72.3 | 17.9 / 63.8 | 22.6 / 68.1 | 23.8 / 71.8 |
| main · emitted · free-form | 106 | 22.5 / 64.2 | 15.2 / 53.8 | 19.6 / 56.6 | 22.3 / 61.3 |
| main · emitted · MC | 107 | 26.2 / 80.4 | 20.6 / 73.8 | 25.6 / 79.4 | 25.4 / 82.2 |
| main · not emitted | 52 | 22.8 / 75.0 | 18.0 / 76.9 | 18.0 / 76.9 | 18.0 / 76.9 |
| main · not emitted · free-form | 9 | 13.9 / 44.4 | 15.3 / 66.7 | 15.3 / 66.7 | 15.3 / 66.7 |
| leaky only | 33 | 20.8 / 66.7 | 17.0 / 66.7 | 21.6 / 78.8 | 21.2 / 72.7 |
| all 300 | 298 | 23.7 / 72.1 | 17.8 / 66.4 | 21.7 / 70.8 | 22.5 / 72.8 |

### 짝 비교 (A − B)

| subset | n | A − B | Δpass@1 (pp) [95% CI] | Δpass@8 (pp) [95% CI] |
|---|---|---|---|---|
| main (non-leaky) | 265 | oracle − text_only | -2.4 [-4.6, -0.1] | -3.0 [-9.1, +3.0] |
| main (non-leaky) | 265 | oracle − corrupt | -1.0 [-2.6, +0.6] | -3.0 [-7.2, +1.1] |
| main (non-leaky) | 265 | oracle − model_drawn | +3.8 [+2.2, +5.4] | +3.4 [-1.1, +7.9] |
| main (non-leaky) | 265 | model_drawn − text_only | -6.1 [-8.5, -3.8] | -6.4 [-12.5, -0.4] |
| main (non-leaky) | 265 | corrupt − text_only | -1.4 [-3.8, +1.1] | +0.0 [-6.0, +6.0] |
| main · free-form | 115 | oracle − text_only | -2.6 [-5.7, +0.4] | -5.2 [-13.9, +3.5] |
| main · free-form | 115 | oracle − corrupt | -2.5 [-5.1, +0.0] | -4.3 [-11.3, +3.5] |
| main · free-form | 115 | oracle − model_drawn | +4.0 [+1.4, +6.7] | +2.6 [-5.2, +10.4] |
| main · free-form | 115 | model_drawn − text_only | -6.6 [-10.0, -3.5] | -7.8 [-16.5, +0.9] |
| main · free-form | 115 | corrupt − text_only | -0.1 [-3.9, +3.7] | -0.9 [-10.4, +8.7] |
| main · MC | 150 | oracle − text_only | -2.2 [-5.3, +1.0] | -1.3 [-9.3, +6.7] |
| main · MC | 150 | oracle − corrupt | +0.2 [-1.8, +2.1] | -2.0 [-6.7, +2.0] |
| main · MC | 150 | oracle − model_drawn | +3.6 [+1.6, +5.6] | +4.0 [-0.7, +8.7] |
| main · MC | 150 | model_drawn − text_only | -5.8 [-9.0, -2.5] | -5.3 [-13.3, +2.7] |
| main · MC | 150 | corrupt − text_only | -2.3 [-5.5, +0.8] | +0.7 [-6.7, +8.0] |
| main · emitted | 213 | oracle − text_only | -1.8 [-4.3, +0.7] | -4.2 [-10.8, +1.9] |
| main · emitted | 213 | oracle − corrupt | -1.2 [-3.2, +0.7] | -3.8 [-8.9, +1.4] |
| main · emitted | 213 | oracle − model_drawn | +4.7 [+2.8, +6.7] | +4.2 [-1.4, +9.9] |
| main · emitted | 213 | model_drawn − text_only | -6.5 [-9.2, -3.9] | -8.5 [-15.0, -2.3] |
| main · emitted | 213 | corrupt − text_only | -0.5 [-3.3, +2.2] | -0.5 [-7.0, +6.1] |
| main · emitted · free-form | 106 | oracle − text_only | -2.9 [-6.2, +0.2] | -7.5 [-17.0, +1.9] |
| main · emitted · free-form | 106 | oracle − corrupt | -2.7 [-5.7, +0.1] | -4.7 [-13.2, +2.8] |
| main · emitted · free-form | 106 | oracle − model_drawn | +4.4 [+1.5, +7.3] | +2.8 [-5.7, +11.3] |
| main · emitted · free-form | 106 | model_drawn − text_only | -7.3 [-10.8, -4.0] | -10.4 [-19.8, -0.9] |
| main · emitted · free-form | 106 | corrupt − text_only | -0.2 [-4.4, +3.8] | -2.8 [-13.2, +7.5] |
| main · emitted · MC | 107 | oracle − text_only | -0.6 [-4.4, +3.3] | -0.9 [-9.3, +7.5] |
| main · emitted · MC | 107 | oracle − corrupt | +0.2 [-2.5, +2.9] | -2.8 [-9.3, +2.8] |
| main · emitted · MC | 107 | oracle − model_drawn | +5.0 [+2.3, +7.9] | +5.6 [-0.9, +12.1] |
| main · emitted · MC | 107 | model_drawn − text_only | -5.6 [-9.6, -1.8] | -6.5 [-15.9, +2.8] |
| main · emitted · MC | 107 | corrupt − text_only | -0.8 [-4.6, +2.9] | +1.9 [-6.5, +10.3] |
| main · not emitted | 52 | oracle − text_only | -4.8 [-9.9, +0.0] | +1.9 [-13.5, +17.3] |
| main · not emitted | 52 | oracle − corrupt | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| main · not emitted | 52 | oracle − model_drawn | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| main · not emitted | 52 | model_drawn − text_only | -4.8 [-9.9, +0.0] | +1.9 [-13.5, +17.3] |
| main · not emitted | 52 | corrupt − text_only | -4.8 [-9.9, +0.0] | +1.9 [-13.5, +17.3] |
| main · not emitted · free-form | 9 | oracle − text_only | +1.4 [-8.3, +9.7] | +22.2 [+0.0, +55.6] |
| main · not emitted · free-form | 9 | oracle − corrupt | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| main · not emitted · free-form | 9 | oracle − model_drawn | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| main · not emitted · free-form | 9 | model_drawn − text_only | +1.4 [-8.3, +9.7] | +22.2 [+0.0, +55.6] |
| main · not emitted · free-form | 9 | corrupt − text_only | +1.4 [-8.3, +9.7] | +22.2 [+0.0, +55.6] |

### <im_start> 를 낸 샘플만 (샘플 단위, 그림 세 조건; prefix 공유)

| subset | n 샘플 | model_drawn | oracle | corrupt | oracle − model_drawn [CI] | oracle − corrupt [CI] |
|---|---|---|---|---|---|---|
| main (non-leaky) | 1077 (213 프롬프트) | 15.3 | 22.7 | 24.7 | +7.4 [+4.5, +10.6] | -1.9 [-5.1, +1.2] |
| main · free-form | 636 (106 프롬프트) | 14.8 | 20.6 | 24.2 | +5.8 [+2.1, +9.6] | -3.6 [-7.5, +0.0] |
| main · MC | 441 (107 프롬프트) | 16.1 | 25.9 | 25.4 | +9.8 [+4.5, +15.1] | +0.5 [-4.6, +5.7] |

### <im_start> 위치 (1200 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 663
- 분포: 0–0: 1175, 1–5: 6, 6–20: 2, 21–50: 1, 51–100: 2, 101–200: 6, 201–768: 8
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 94.8 | 7.7 | 0.00 | 0.0 | 276 |
| model_drawn | 76.2 | 7.7 | 0.32 | 0.0 | 293 |
| oracle | 93.9 | 9.5 | 0.51 | 0.5 | 366 |
| corrupt | 94.0 | 9.1 | 0.51 | 0.5 | 366 |
