# Stage 1 — oracle on demand (/tmp/claude-1001/-mnt-work-tar-rlvr/76c28d1e-c57e-474f-8165-2a16be721a6f/scratchpad/clutrr_greedy)

프롬프트 200 (held-out 0), 객관식 0, leaky (휴리스틱) 7 → main = leaky 제외 193.


## greedy (1 샘플)

<im_start> 를 낸 프롬프트: 133/200


### 조건별 greedy acc (%, 95% CI)

| subset | n | text_only | model_drawn | oracle_S0 | corrupt_S0 |
|---|---|---|---|---|---|
| main (non-leaky) | 193 | 13.5 [9.4, 19.0] | 14.5 [10.2, 20.2] | 9.8 [6.4, 14.9] | 13.0 [8.9, 18.4] |
| main · free-form | 193 | 13.5 [9.4, 19.0] | 14.5 [10.2, 20.2] | 9.8 [6.4, 14.9] | 13.0 [8.9, 18.4] |
| main · emitted | 128 | 15.6 [10.3, 22.9] | 13.3 [8.5, 20.2] | 6.2 [3.2, 11.8] | 10.9 [6.6, 17.5] |
| main · emitted · free-form | 128 | 15.6 [10.3, 22.9] | 13.3 [8.5, 20.2] | 6.2 [3.2, 11.8] | 10.9 [6.6, 17.5] |
| main · not emitted | 65 | 9.2 [4.3, 18.7] | 16.9 [9.7, 27.8] | 16.9 [9.7, 27.8] | 16.9 [9.7, 27.8] |
| main · not emitted · free-form | 65 | 9.2 [4.3, 18.7] | 16.9 [9.7, 27.8] | 16.9 [9.7, 27.8] | 16.9 [9.7, 27.8] |
| leaky only | 7 | 14.3 [2.6, 51.3] | 0.0 [0.0, 35.4] | 0.0 [0.0, 35.4] | 0.0 [0.0, 35.4] |
| all 300 | 200 | 13.5 [9.4, 18.9] | 14.0 [9.9, 19.5] | 9.5 [6.2, 14.4] | 12.5 [8.6, 17.8] |

### 짝 비교 (A − B)

| subset | n | A − B | diff (pp) | A만 맞음 / B만 맞음 | McNemar p |
|---|---|---|---|---|---|
| main (non-leaky) | 193 | oracle_S0 − text_only | -3.6 | 14 / 21 | 0.311 |
| main (non-leaky) | 193 | model_drawn − text_only | +1.0 | 21 / 19 | 0.875 |
| main (non-leaky) | 193 | oracle_S0 − corrupt_S0 | -3.1 | 3 / 9 | 0.146 |
| main (non-leaky) | 193 | corrupt_S0 − text_only | -0.5 | 19 / 20 | 1 |
| main · free-form | 193 | oracle_S0 − text_only | -3.6 | 14 / 21 | 0.311 |
| main · free-form | 193 | model_drawn − text_only | +1.0 | 21 / 19 | 0.875 |
| main · free-form | 193 | oracle_S0 − corrupt_S0 | -3.1 | 3 / 9 | 0.146 |
| main · free-form | 193 | corrupt_S0 − text_only | -0.5 | 19 / 20 | 1 |
| main · emitted | 128 | oracle_S0 − text_only | -9.4 | 5 / 17 | 0.0169 |
| main · emitted | 128 | model_drawn − text_only | -2.3 | 12 / 15 | 0.701 |
| main · emitted | 128 | oracle_S0 − corrupt_S0 | -4.7 | 3 / 9 | 0.146 |
| main · emitted | 128 | corrupt_S0 − text_only | -4.7 | 10 / 16 | 0.327 |
| main · emitted · free-form | 128 | oracle_S0 − text_only | -9.4 | 5 / 17 | 0.0169 |
| main · emitted · free-form | 128 | model_drawn − text_only | -2.3 | 12 / 15 | 0.701 |
| main · emitted · free-form | 128 | oracle_S0 − corrupt_S0 | -4.7 | 3 / 9 | 0.146 |
| main · emitted · free-form | 128 | corrupt_S0 − text_only | -4.7 | 10 / 16 | 0.327 |
| main · not emitted | 65 | oracle_S0 − text_only | +7.7 | 9 / 4 | 0.267 |
| main · not emitted | 65 | model_drawn − text_only | +7.7 | 9 / 4 | 0.267 |
| main · not emitted | 65 | oracle_S0 − corrupt_S0 | +0.0 | 0 / 0 | 1 |
| main · not emitted | 65 | corrupt_S0 − text_only | +7.7 | 9 / 4 | 0.267 |
| main · not emitted · free-form | 65 | oracle_S0 − text_only | +7.7 | 9 / 4 | 0.267 |
| main · not emitted · free-form | 65 | model_drawn − text_only | +7.7 | 9 / 4 | 0.267 |
| main · not emitted · free-form | 65 | oracle_S0 − corrupt_S0 | +0.0 | 0 / 0 | 1 |
| main · not emitted · free-form | 65 | corrupt_S0 − text_only | +7.7 | 9 / 4 | 0.267 |

### <im_start> 위치 (133 샘플)

- completion 시작부터 토큰 수: 중앙값 0, 사분위 [0, 0], 최소 0, 최대 0
- 분포: 0–0: 133, 1–5: 0, 6–20: 0, 21–50: 0, 51–100: 0, 101–200: 0, 201–768: 0
- model_drawn 최종 길이 대비 상대 위치: 중앙값 0.00, 사분위 [0.00, 0.00]

### 행동 지표 (main)

| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |
|---|---|---|---|---|---|
| text_only | 77.7 | 3.6 | 0.00 | 0.0 | 50 |
| model_drawn | 66.8 | 3.1 | 0.56 | 0.0 | 197 |
| oracle_S0 | 60.1 | 26.4 | 0.66 | 0.0 | 837 |
| corrupt_S0 | 64.2 | 25.4 | 0.66 | 0.0 | 837 |
