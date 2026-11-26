# Stage 1 — oracle on demand (logs/stage1/b1536)

프롬프트 298 (held-out 100), 객관식 162, leaky (휴리스틱) 33 → main = leaky 제외 265.


## greedy (1 샘플)

<im_start> 를 낸 프롬프트: 185/298


### 조건별 greedy acc (%, 95% CI)

| subset | n | text_only | model_drawn | model_drawn_S0 | oracle_S2 | oracle_S0 | corrupt |
|---|---|---|---|---|---|---|---|
| main (non-leaky) | 265 | 31.3 [26.0, 37.1] | 21.9 [17.3, 27.2] | 20.0 [15.6, 25.2] | 24.2 [19.4, 29.6] | 22.6 [18.0, 28.1] | 22.6 [18.0, 28.1] |
| main · free-form | 115 | 28.7 [21.2, 37.5] | 21.7 [15.2, 30.1] | 16.5 [10.8, 24.4] | 24.3 [17.4, 32.9] | 19.1 [13.0, 27.3] | 24.3 [17.4, 32.9] |
| main · MC | 150 | 33.3 [26.3, 41.2] | 22.0 [16.1, 29.3] | 22.7 [16.7, 30.0] | 24.0 [17.9, 31.4] | 25.3 [19.0, 32.8] | 21.3 [15.5, 28.6] |
| main · emitted | 167 | 34.1 [27.4, 41.6] | 23.4 [17.6, 30.3] | 20.4 [15.0, 27.1] | 26.9 [20.8, 34.1] | 24.6 [18.6, 31.6] | 24.6 [18.6, 31.6] |
| main · emitted · free-form | 89 | 32.6 [23.7, 42.9] | 23.6 [16.0, 33.4] | 16.9 [10.5, 26.0] | 27.0 [18.8, 37.0] | 20.2 [13.2, 29.7] | 27.0 [18.8, 37.0] |
| main · emitted · MC | 78 | 35.9 [26.1, 47.0] | 23.1 [15.1, 33.6] | 24.4 [16.2, 34.9] | 26.9 [18.3, 37.7] | 29.5 [20.5, 40.4] | 21.8 [14.1, 32.2] |
| main · not emitted | 98 | 26.5 [18.8, 36.0] | 19.4 [12.8, 28.3] | 19.4 [12.8, 28.3] | 19.4 [12.8, 28.3] | 19.4 [12.8, 28.3] | 19.4 [12.8, 28.3] |
| main · not emitted · free-form | 26 | 15.4 [6.1, 33.5] | 15.4 [6.1, 33.5] | 15.4 [6.1, 33.5] | 15.4 [6.1, 33.5] | 15.4 [6.1, 33.5] | 15.4 [6.1, 33.5] |
| leaky only | 33 | 42.4 [27.2, 59.2] | 21.2 [10.7, 37.8] | 21.2 [10.7, 37.8] | 36.4 [22.2, 53.4] | 24.2 [12.8, 41.0] | 27.3 [15.1, 44.2] |
| all 300 | 298 | 32.6 [27.5, 38.1] | 21.8 [17.5, 26.8] | 20.1 [16.0, 25.1] | 25.5 [20.9, 30.7] | 22.8 [18.4, 27.9] | 23.2 [18.7, 28.3] |

### 짝 비교 (A − B)

| subset | n | A − B | diff (pp) | A만 맞음 / B만 맞음 | McNemar p |
|---|---|---|---|---|---|
| main (non-leaky) | 265 | oracle_S2 − text_only | -7.2 | 33 / 52 | 0.0503 |
| main (non-leaky) | 265 | oracle_S2 − corrupt | +1.5 | 23 / 19 | 0.644 |
| main (non-leaky) | 265 | oracle_S2 − model_drawn | +2.3 | 21 / 15 | 0.405 |
| main (non-leaky) | 265 | oracle_S0 − text_only | -8.7 | 26 / 49 | 0.0106 |
| main (non-leaky) | 265 | oracle_S0 − corrupt | +0.0 | 22 / 22 | 1 |
| main (non-leaky) | 265 | oracle_S0 − oracle_S2 | -1.5 | 24 / 28 | 0.678 |
| main (non-leaky) | 265 | oracle_S0 − model_drawn_S0 | +2.6 | 28 / 21 | 0.392 |
| main (non-leaky) | 265 | model_drawn_S0 − model_drawn | -1.9 | 18 / 23 | 0.533 |
| main (non-leaky) | 265 | model_drawn − text_only | -9.4 | 27 / 52 | 0.00655 |
| main (non-leaky) | 265 | model_drawn_S0 − text_only | -11.3 | 27 / 57 | 0.0014 |
| main (non-leaky) | 265 | corrupt − text_only | -8.7 | 28 / 51 | 0.0128 |
| main · free-form | 115 | oracle_S2 − text_only | -4.3 | 14 / 19 | 0.487 |
| main · free-form | 115 | oracle_S2 − corrupt | +0.0 | 9 / 9 | 1 |
| main · free-form | 115 | oracle_S2 − model_drawn | +2.6 | 10 / 7 | 0.629 |
| main · free-form | 115 | oracle_S0 − text_only | -9.6 | 9 / 20 | 0.0614 |
| main · free-form | 115 | oracle_S0 − corrupt | -5.2 | 5 / 11 | 0.21 |
| main · free-form | 115 | oracle_S0 − oracle_S2 | -5.2 | 12 / 18 | 0.362 |
| main · free-form | 115 | oracle_S0 − model_drawn_S0 | +2.6 | 13 / 10 | 0.678 |
| main · free-form | 115 | model_drawn_S0 − model_drawn | -5.2 | 7 / 13 | 0.263 |
| main · free-form | 115 | model_drawn − text_only | -7.0 | 11 / 19 | 0.2 |
| main · free-form | 115 | model_drawn_S0 − text_only | -12.2 | 8 / 22 | 0.0161 |
| main · free-form | 115 | corrupt − text_only | -4.3 | 11 / 16 | 0.442 |
| main · MC | 150 | oracle_S2 − text_only | -9.3 | 19 / 33 | 0.0704 |
| main · MC | 150 | oracle_S2 − corrupt | +2.7 | 14 / 10 | 0.541 |
| main · MC | 150 | oracle_S2 − model_drawn | +2.0 | 11 / 8 | 0.648 |
| main · MC | 150 | oracle_S0 − text_only | -8.0 | 17 / 29 | 0.104 |
| main · MC | 150 | oracle_S0 − corrupt | +4.0 | 17 / 11 | 0.345 |
| main · MC | 150 | oracle_S0 − oracle_S2 | +1.3 | 12 / 10 | 0.832 |
| main · MC | 150 | oracle_S0 − model_drawn_S0 | +2.7 | 15 / 11 | 0.557 |
| main · MC | 150 | model_drawn_S0 − model_drawn | +0.7 | 11 / 10 | 1 |
| main · MC | 150 | model_drawn − text_only | -11.3 | 16 / 33 | 0.0213 |
| main · MC | 150 | model_drawn_S0 − text_only | -10.7 | 19 / 35 | 0.0402 |
| main · MC | 150 | corrupt − text_only | -12.0 | 17 / 35 | 0.0175 |
| main · emitted | 167 | oracle_S2 − text_only | -7.2 | 24 / 36 | 0.155 |
| main · emitted | 167 | oracle_S2 − corrupt | +2.4 | 23 / 19 | 0.644 |
| main · emitted | 167 | oracle_S2 − model_drawn | +3.6 | 21 / 15 | 0.405 |
| main · emitted | 167 | oracle_S0 − text_only | -9.6 | 17 / 33 | 0.0328 |
| main · emitted | 167 | oracle_S0 − corrupt | +0.0 | 22 / 22 | 1 |
| main · emitted | 167 | oracle_S0 − oracle_S2 | -2.4 | 24 / 28 | 0.678 |
| main · emitted | 167 | oracle_S0 − model_drawn_S0 | +4.2 | 28 / 21 | 0.392 |
| main · emitted | 167 | model_drawn_S0 − model_drawn | -3.0 | 18 / 23 | 0.533 |
| main · emitted | 167 | model_drawn − text_only | -10.8 | 18 / 36 | 0.0198 |
| main · emitted | 167 | model_drawn_S0 − text_only | -13.8 | 18 / 41 | 0.00379 |
| main · emitted | 167 | corrupt − text_only | -9.6 | 19 / 35 | 0.0402 |
| main · emitted · free-form | 89 | oracle_S2 − text_only | -5.6 | 13 / 18 | 0.473 |
| main · emitted · free-form | 89 | oracle_S2 − corrupt | +0.0 | 9 / 9 | 1 |
| main · emitted · free-form | 89 | oracle_S2 − model_drawn | +3.4 | 10 / 7 | 0.629 |
| main · emitted · free-form | 89 | oracle_S0 − text_only | -12.4 | 8 / 19 | 0.0522 |
| main · emitted · free-form | 89 | oracle_S0 − corrupt | -6.7 | 5 / 11 | 0.21 |
| main · emitted · free-form | 89 | oracle_S0 − oracle_S2 | -6.7 | 12 / 18 | 0.362 |
| main · emitted · free-form | 89 | oracle_S0 − model_drawn_S0 | +3.4 | 13 / 10 | 0.678 |
| main · emitted · free-form | 89 | model_drawn_S0 − model_drawn | -6.7 | 7 / 13 | 0.263 |
| main · emitted · free-form | 89 | model_drawn − text_only | -9.0 | 10 / 18 | 0.185 |
| main · emitted · free-form | 89 | model_drawn_S0 − text_only | -15.7 | 7 / 21 | 0.0125 |
| main · emitted · free-form | 89 | corrupt − text_only | -5.6 | 10 / 15 | 0.424 |
| main · emitted · MC | 78 | oracle_S2 − text_only | -9.0 | 11 / 18 | 0.265 |
| main · emitted · MC | 78 | oracle_S2 − corrupt | +5.1 | 14 / 10 | 0.541 |
| main · emitted · MC | 78 | oracle_S2 − model_drawn | +3.8 | 11 / 8 | 0.648 |
| main · emitted · MC | 78 | oracle_S0 − text_only | -6.4 | 9 / 14 | 0.405 |
| main · emitted · MC | 78 | oracle_S0 − corrupt | +7.7 | 17 / 11 | 0.345 |
| main · emitted · MC | 78 | oracle_S0 − oracle_S2 | +2.6 | 12 / 10 | 0.832 |
| main · emitted · MC | 78 | oracle_S0 − model_drawn_S0 | +5.1 | 15 / 11 | 0.557 |
| main · emitted · MC | 78 | model_drawn_S0 − model_drawn | +1.3 | 11 / 10 | 1 |
| main · emitted · MC | 78 | model_drawn − text_only | -12.8 | 8 / 18 | 0.0755 |
| main · emitted · MC | 78 | model_drawn_S0 − text_only | -11.5 | 11 / 20 | 0.15 |
| main · emitted · MC | 78 | corrupt − text_only | -14.1 | 9 / 20 | 0.0614 |
| main · not emitted | 98 | oracle_S2 − text_only | -7.1 | 9 / 16 | 0.23 |
| main · not emitted | 98 | oracle_S2 − corrupt | +0.0 | 0 / 0 | 1 |
| main · not emitted | 98 | oracle_S2 − model_drawn | +0.0 | 0 / 0 | 1 |
| main · not emitted | 98 | oracle_S0 − text_only | -7.1 | 9 / 16 | 0.23 |
| main · not emitted | 98 | oracle_S0 − corrupt | +0.0 | 0 / 0 | 1 |
| main · not emitted | 98 | oracle_S0 − oracle_S2 | +0.0 | 0 / 0 | 1 |
| main · not emitted | 98 | oracle_S0 − model_drawn_S0 | +0.0 | 0 / 0 | 1 |
| main · not emitted | 98 | model_drawn_S0 − model_drawn | +0.0 | 0 / 0 | 1 |
| main · not emitted | 98 | model_drawn − text_only | -7.1 | 9 / 16 | 0.23 |
| main · not emitted | 98 | model_drawn_S0 − text_only | -7.1 | 9 / 16 | 0.23 |
| main · not emitted | 98 | corrupt − text_only | -7.1 | 9 / 16 | 0.23 |
| main · not emitted · free-form | 26 | oracle_S2 − text_only | +0.0 | 1 / 1 | 1 |
| main · not emitted · free-form | 26 | oracle_S2 − corrupt | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | oracle_S2 − model_drawn | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | oracle_S0 − text_only | +0.0 | 1 / 1 | 1 |
| main · not emitted · free-form | 26 | oracle_S0 − corrupt | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | oracle_S0 − oracle_S2 | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | oracle_S0 − model_drawn_S0 | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | model_drawn_S0 − model_drawn | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 26 | model_drawn − text_only | +0.0 | 1 / 1 | 1 |
| main · not emitted · free-form | 26 | model_drawn_S0 − text_only | +0.0 | 1 / 1 | 1 |
| main · not emitted · free-form | 26 | corrupt − text_only | +0.0 | 1 / 1 | 1 |

### <im_start> 위치 (185 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 152
- 분포: 0–0: 180, 1–5: 0, 6–20: 0, 21–50: 2, 51–100: 1, 101–200: 2, 201–768: 0
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 93.6 | 7.9 | 0.00 | 0.0 | 354 |
| model_drawn | 94.7 | 7.5 | 0.62 | 0.0 | 375 |
| model_drawn_S0 | 87.5 | 15.1 | 0.65 | 1.9 | 911 |
| oracle_S2 | 96.2 | 8.3 | 0.63 | 0.0 | 380 |
| oracle_S0 | 91.7 | 14.3 | 0.63 | 0.4 | 881 |
| corrupt | 95.5 | 8.3 | 0.63 | 0.4 | 370 |
