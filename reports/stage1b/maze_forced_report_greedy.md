# Stage 1 — oracle on demand (logs/stage1b/maze_forced)

프롬프트 200 (held-out 0), 객관식 0, leaky (휴리스틱) 0 → main = leaky 제외 200.


## greedy (1 샘플)

<im_start> 를 낸 프롬프트: 200/200


### 조건별 greedy acc (%, 95% CI)

| subset | n | text_only | model_drawn | model_drawn_S0 | oracle_S2 | oracle_S0 | corrupt | corrupt_S0 |
|---|---|---|---|---|---|---|---|---|
| main (non-leaky) | 200 | 13.5 [9.4, 18.9] | 35.0 [28.7, 41.8] | 27.0 [21.3, 33.5] | 29.0 [23.2, 35.6] | 22.0 [16.8, 28.2] | 30.5 [24.5, 37.2] | 23.5 [18.2, 29.8] |
| main · free-form | 200 | 13.5 [9.4, 18.9] | 35.0 [28.7, 41.8] | 27.0 [21.3, 33.5] | 29.0 [23.2, 35.6] | 22.0 [16.8, 28.2] | 30.5 [24.5, 37.2] | 23.5 [18.2, 29.8] |
| main · emitted | 200 | 13.5 [9.4, 18.9] | 35.0 [28.7, 41.8] | 27.0 [21.3, 33.5] | 29.0 [23.2, 35.6] | 22.0 [16.8, 28.2] | 30.5 [24.5, 37.2] | 23.5 [18.2, 29.8] |
| main · emitted · free-form | 200 | 13.5 [9.4, 18.9] | 35.0 [28.7, 41.8] | 27.0 [21.3, 33.5] | 29.0 [23.2, 35.6] | 22.0 [16.8, 28.2] | 30.5 [24.5, 37.2] | 23.5 [18.2, 29.8] |
| all 300 | 200 | 13.5 [9.4, 18.9] | 35.0 [28.7, 41.8] | 27.0 [21.3, 33.5] | 29.0 [23.2, 35.6] | 22.0 [16.8, 28.2] | 30.5 [24.5, 37.2] | 23.5 [18.2, 29.8] |
| main · solvable | 109 | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] |
| main · UNSOLVABLE | 91 | 29.7 [21.3, 39.7] | 76.9 [67.3, 84.4] | 59.3 [49.1, 68.9] | 63.7 [53.5, 72.9] | 48.4 [38.4, 58.5] | 67.0 [56.9, 75.8] | 51.6 [41.5, 61.6] |
| main · emitted · solvable | 109 | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] | 0.0 [0.0, 3.4] |

### 짝 비교 (A − B)

| subset | n | A − B | diff (pp) | A만 맞음 / B만 맞음 | McNemar p |
|---|---|---|---|---|---|
| main (non-leaky) | 200 | oracle_S2 − text_only | +15.5 | 38 / 7 | 3.12e-06 |
| main (non-leaky) | 200 | oracle_S2 − corrupt | -1.5 | 20 / 23 | 0.761 |
| main (non-leaky) | 200 | oracle_S2 − model_drawn | -6.0 | 13 / 25 | 0.073 |
| main (non-leaky) | 200 | oracle_S0 − text_only | +8.5 | 34 / 17 | 0.0241 |
| main (non-leaky) | 200 | oracle_S0 − corrupt | -8.5 | 12 / 29 | 0.0115 |
| main (non-leaky) | 200 | oracle_S0 − oracle_S2 | -7.0 | 18 / 32 | 0.0649 |
| main (non-leaky) | 200 | oracle_S0 − model_drawn_S0 | -5.0 | 20 / 30 | 0.203 |
| main (non-leaky) | 200 | model_drawn_S0 − model_drawn | -8.0 | 11 / 27 | 0.0139 |
| main (non-leaky) | 200 | model_drawn − text_only | +21.5 | 51 / 8 | 9.05e-09 |
| main (non-leaky) | 200 | model_drawn_S0 − text_only | +13.5 | 40 / 13 | 0.000269 |
| main (non-leaky) | 200 | corrupt − text_only | +17.0 | 44 / 10 | 3.39e-06 |
| main (non-leaky) | 200 | oracle_S0 − corrupt_S0 | -1.5 | 19 / 22 | 0.755 |
| main (non-leaky) | 200 | corrupt_S0 − text_only | +10.0 | 37 / 17 | 0.00907 |
| main · free-form | 200 | oracle_S2 − text_only | +15.5 | 38 / 7 | 3.12e-06 |
| main · free-form | 200 | oracle_S2 − corrupt | -1.5 | 20 / 23 | 0.761 |
| main · free-form | 200 | oracle_S2 − model_drawn | -6.0 | 13 / 25 | 0.073 |
| main · free-form | 200 | oracle_S0 − text_only | +8.5 | 34 / 17 | 0.0241 |
| main · free-form | 200 | oracle_S0 − corrupt | -8.5 | 12 / 29 | 0.0115 |
| main · free-form | 200 | oracle_S0 − oracle_S2 | -7.0 | 18 / 32 | 0.0649 |
| main · free-form | 200 | oracle_S0 − model_drawn_S0 | -5.0 | 20 / 30 | 0.203 |
| main · free-form | 200 | model_drawn_S0 − model_drawn | -8.0 | 11 / 27 | 0.0139 |
| main · free-form | 200 | model_drawn − text_only | +21.5 | 51 / 8 | 9.05e-09 |
| main · free-form | 200 | model_drawn_S0 − text_only | +13.5 | 40 / 13 | 0.000269 |
| main · free-form | 200 | corrupt − text_only | +17.0 | 44 / 10 | 3.39e-06 |
| main · free-form | 200 | oracle_S0 − corrupt_S0 | -1.5 | 19 / 22 | 0.755 |
| main · free-form | 200 | corrupt_S0 − text_only | +10.0 | 37 / 17 | 0.00907 |
| main · emitted | 200 | oracle_S2 − text_only | +15.5 | 38 / 7 | 3.12e-06 |
| main · emitted | 200 | oracle_S2 − corrupt | -1.5 | 20 / 23 | 0.761 |
| main · emitted | 200 | oracle_S2 − model_drawn | -6.0 | 13 / 25 | 0.073 |
| main · emitted | 200 | oracle_S0 − text_only | +8.5 | 34 / 17 | 0.0241 |
| main · emitted | 200 | oracle_S0 − corrupt | -8.5 | 12 / 29 | 0.0115 |
| main · emitted | 200 | oracle_S0 − oracle_S2 | -7.0 | 18 / 32 | 0.0649 |
| main · emitted | 200 | oracle_S0 − model_drawn_S0 | -5.0 | 20 / 30 | 0.203 |
| main · emitted | 200 | model_drawn_S0 − model_drawn | -8.0 | 11 / 27 | 0.0139 |
| main · emitted | 200 | model_drawn − text_only | +21.5 | 51 / 8 | 9.05e-09 |
| main · emitted | 200 | model_drawn_S0 − text_only | +13.5 | 40 / 13 | 0.000269 |
| main · emitted | 200 | corrupt − text_only | +17.0 | 44 / 10 | 3.39e-06 |
| main · emitted | 200 | oracle_S0 − corrupt_S0 | -1.5 | 19 / 22 | 0.755 |
| main · emitted | 200 | corrupt_S0 − text_only | +10.0 | 37 / 17 | 0.00907 |
| main · emitted · free-form | 200 | oracle_S2 − text_only | +15.5 | 38 / 7 | 3.12e-06 |
| main · emitted · free-form | 200 | oracle_S2 − corrupt | -1.5 | 20 / 23 | 0.761 |
| main · emitted · free-form | 200 | oracle_S2 − model_drawn | -6.0 | 13 / 25 | 0.073 |
| main · emitted · free-form | 200 | oracle_S0 − text_only | +8.5 | 34 / 17 | 0.0241 |
| main · emitted · free-form | 200 | oracle_S0 − corrupt | -8.5 | 12 / 29 | 0.0115 |
| main · emitted · free-form | 200 | oracle_S0 − oracle_S2 | -7.0 | 18 / 32 | 0.0649 |
| main · emitted · free-form | 200 | oracle_S0 − model_drawn_S0 | -5.0 | 20 / 30 | 0.203 |
| main · emitted · free-form | 200 | model_drawn_S0 − model_drawn | -8.0 | 11 / 27 | 0.0139 |
| main · emitted · free-form | 200 | model_drawn − text_only | +21.5 | 51 / 8 | 9.05e-09 |
| main · emitted · free-form | 200 | model_drawn_S0 − text_only | +13.5 | 40 / 13 | 0.000269 |
| main · emitted · free-form | 200 | corrupt − text_only | +17.0 | 44 / 10 | 3.39e-06 |
| main · emitted · free-form | 200 | oracle_S0 − corrupt_S0 | -1.5 | 19 / 22 | 0.755 |
| main · emitted · free-form | 200 | corrupt_S0 − text_only | +10.0 | 37 / 17 | 0.00907 |
| main · solvable | 109 | oracle_S2 − text_only | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S2 − corrupt | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S2 − model_drawn | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S0 − text_only | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S0 − corrupt | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S0 − oracle_S2 | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S0 − model_drawn_S0 | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | model_drawn_S0 − model_drawn | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | model_drawn − text_only | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | model_drawn_S0 − text_only | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | corrupt − text_only | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | oracle_S0 − corrupt_S0 | +0.0 | 0 / 0 | 1 |
| main · solvable | 109 | corrupt_S0 − text_only | +0.0 | 0 / 0 | 1 |
| main · UNSOLVABLE | 91 | oracle_S2 − text_only | +34.1 | 38 / 7 | 3.12e-06 |
| main · UNSOLVABLE | 91 | oracle_S2 − corrupt | -3.3 | 20 / 23 | 0.761 |
| main · UNSOLVABLE | 91 | oracle_S2 − model_drawn | -13.2 | 13 / 25 | 0.073 |
| main · UNSOLVABLE | 91 | oracle_S0 − text_only | +18.7 | 34 / 17 | 0.0241 |
| main · UNSOLVABLE | 91 | oracle_S0 − corrupt | -18.7 | 12 / 29 | 0.0115 |
| main · UNSOLVABLE | 91 | oracle_S0 − oracle_S2 | -15.4 | 18 / 32 | 0.0649 |
| main · UNSOLVABLE | 91 | oracle_S0 − model_drawn_S0 | -11.0 | 20 / 30 | 0.203 |
| main · UNSOLVABLE | 91 | model_drawn_S0 − model_drawn | -17.6 | 11 / 27 | 0.0139 |
| main · UNSOLVABLE | 91 | model_drawn − text_only | +47.3 | 51 / 8 | 9.05e-09 |
| main · UNSOLVABLE | 91 | model_drawn_S0 − text_only | +29.7 | 40 / 13 | 0.000269 |
| main · UNSOLVABLE | 91 | corrupt − text_only | +37.4 | 44 / 10 | 3.39e-06 |
| main · UNSOLVABLE | 91 | oracle_S0 − corrupt_S0 | -3.3 | 19 / 22 | 0.755 |
| main · UNSOLVABLE | 91 | corrupt_S0 − text_only | +22.0 | 37 / 17 | 0.00907 |

### <im_start> 위치 (200 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 0
- 분포: 0–0: 200, 1–5: 0, 6–20: 0, 21–50: 0, 51–100: 0, 101–200: 0, 201–768: 0
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 53.0 | 64.0 | 0.00 | 0.0 | 1536 |
| model_drawn | 89.0 | 42.0 | 1.00 | 0.0 | 279 |
| model_drawn_S0 | 55.5 | 9.5 | 1.00 | 0.0 | 826 |
| oracle_S2 | 70.0 | 29.0 | 1.00 | 0.0 | 177 |
| oracle_S0 | 43.0 | 44.0 | 1.00 | 0.5 | 886 |
| corrupt | 66.5 | 33.5 | 1.00 | 0.0 | 177 |
| corrupt_S0 | 46.5 | 44.0 | 1.00 | 0.5 | 917 |
