# r06 길이 편향 분석 (sequence-mean 하의 토큰당 gradient 가중치)

스크립트: `scripts/analysis/length_bias.py --run r06-sketch-bridge` (r07 에도 같은 명령으로).

출처: r06 은 전 샘플 덤프가 없어 두 개를 합쳤다.
- reward / advantage: 디버그 로그 `logs/r06-sketch-bridge.log` 의 샘플별 값 (정확, 소수 3자리).
- 그림 샘플 길이: `logs/visual_samples.jsonl` 원문 재토큰화 (2,093 개 전부 매칭, 정확).
- 텍스트 샘플 길이: 로그에서 안 잘린 311 개는 정확, 가운데가 잘린 396 개는 보이는 부분의 토큰/문자 비율로 추정.
- 범위 step 0–174 (2,800 샘플). step 175 는 SIGTERM 으로 로그가 끊겨 없음.
- 검증: step 별 reward_mean 은 metrics.jsonl 과 175 step 모두 일치, step 별 평균 길이 차이 −2.2 토큰.
- bridge 는 18 토큰 (15 아님). w = |adv| / L.


### r06-sketch-bridge: all steps 0–174

샘플 2800 (그림 2093, 텍스트 707); 텍스트 길이 추정 (로그에서 가운데 잘림) 396

Table 1 — completion length (tokens): median / mean (n)

| | correct | incorrect |
|---|---|---|
| image | 354 / 372.9 (512) | 378 / 407.4 (1581) |
| text | 436 / 425.4 (123) | 432 / 419.9 (584) |

image responses — segment means (tokens), mean (median):

| | n | text before image | visual block(s) | bridge | text after bridge | images/resp | with bridge |
|---|---|---|---|---|---|---|---|
| correct | 512 | 2.1 (0) | 114.7 (84) | 17.9 (18) | 238.2 (225) | 1.02 | 0.99 |
| incorrect | 1581 | 2.4 (0) | 124.6 (84) | 17.6 (18) | 262.8 (236) | 1.04 | 0.98 |
| all | 2093 | 2.3 (0) | 122.2 (84) | 17.7 (18) | 256.8 (233) | 1.03 | 0.98 |

Table 2 — mean w = |adv| / length  (×1e-3), n; and mean |adv|

| | correct | incorrect |
|---|---|---|
| image | 3.371 (n=512, |adv| 1.114) | 1.041 (n=1581, |adv| 0.359) |
| text | 14.670 (n=123, |adv| 1.490) | 4.062 (n=584, |adv| 0.318) |

w_text_incorrect / w_image_incorrect = 3.90 (all incorrect); 3.80 (incorrect with adv≠0: text n=358, image n=943)

truncated at max length (len >= 766):

| | n truncated | of | rate |
|---|---|---|---|
| image | 55 | 2093 | 0.026 |
| text | 48 | 707 | 0.068 |

metrics.jsonl length/truncation_rate (토큰 id, EOS 없음 & n >= 767): 합계 163 / 2800; text = 합계 − image(55) = 108 / 707 (0.153)

### r06-sketch-bridge: steps 0–39

샘플 640 (그림 490, 텍스트 150); 텍스트 길이 추정 (로그에서 가운데 잘림) 82

Table 1 — completion length (tokens): median / mean (n)

| | correct | incorrect |
|---|---|---|
| image | 339 / 358.9 (133) | 350 / 396.7 (357) |
| text | 542 / 455.5 (32) | 382 / 396.9 (118) |

image responses — segment means (tokens), mean (median):

| | n | text before image | visual block(s) | bridge | text after bridge | images/resp | with bridge |
|---|---|---|---|---|---|---|---|
| correct | 133 | 0.1 (0) | 116.9 (84) | 18.0 (18) | 223.9 (205) | 1.02 | 1.00 |
| incorrect | 357 | 1.1 (0) | 122.7 (84) | 17.8 (18) | 255.1 (225) | 1.04 | 0.99 |
| all | 490 | 0.8 (0) | 121.2 (84) | 17.9 (18) | 246.6 (218) | 1.03 | 0.99 |

Table 2 — mean w = |adv| / length  (×1e-3), n; and mean |adv|

| | correct | incorrect |
|---|---|---|
| image | 3.293 (n=133, |adv| 1.079) | 1.206 (n=357, |adv| 0.395) |
| text | 8.545 (n=32, |adv| 1.318) | 9.755 (n=118, |adv| 0.379) |

w_text_incorrect / w_image_incorrect = 8.09 (all incorrect); 8.41 (incorrect with adv≠0: text n=76, image n=239)

truncated at max length (len >= 766):

| | n truncated | of | rate |
|---|---|---|---|
| image | 9 | 490 | 0.018 |
| text | 11 | 150 | 0.073 |

metrics.jsonl length/truncation_rate (토큰 id, EOS 없음 & n >= 767): 합계 34 / 640; text = 합계 − image(9) = 25 / 150 (0.167)

### r06-sketch-bridge: steps 40–174

샘플 2160 (그림 1603, 텍스트 557); 텍스트 길이 추정 (로그에서 가운데 잘림) 314

Table 1 — completion length (tokens): median / mean (n)

| | correct | incorrect |
|---|---|---|
| image | 358 / 377.8 (379) | 384 / 410.6 (1224) |
| text | 426 / 414.8 (91) | 440 / 425.7 (466) |

image responses — segment means (tokens), mean (median):

| | n | text before image | visual block(s) | bridge | text after bridge | images/resp | with bridge |
|---|---|---|---|---|---|---|---|
| correct | 379 | 2.8 (0) | 114.0 (84) | 17.9 (18) | 243.2 (230) | 1.02 | 0.99 |
| incorrect | 1224 | 2.8 (0) | 125.2 (84) | 17.6 (18) | 265.1 (243) | 1.04 | 0.98 |
| all | 1603 | 2.8 (0) | 122.5 (84) | 17.7 (18) | 259.9 (239) | 1.03 | 0.98 |

Table 2 — mean w = |adv| / length  (×1e-3), n; and mean |adv|

| | correct | incorrect |
|---|---|---|
| image | 3.398 (n=379, |adv| 1.126) | 0.993 (n=1224, |adv| 0.349) |
| text | 16.824 (n=91, |adv| 1.550) | 2.621 (n=466, |adv| 0.302) |

w_text_incorrect / w_image_incorrect = 2.64 (all incorrect); 2.51 (incorrect with adv≠0: text n=282, image n=704)

truncated at max length (len >= 766):

| | n truncated | of | rate |
|---|---|---|---|
| image | 46 | 1603 | 0.029 |
| text | 37 | 557 | 0.066 |

metrics.jsonl length/truncation_rate (토큰 id, EOS 없음 & n >= 767): 합계 129 / 2160; text = 합계 − image(46) = 83 / 557 (0.149)
