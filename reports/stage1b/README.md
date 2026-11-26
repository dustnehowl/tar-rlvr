# Stage 1b — 비수학 태스크 oracle: maze·clutrr 모두 oracle 효과 없음

Base Tar-7B (LoRA 없음). Stage 1 (math) 과 같은 `scripts/oracle_eval.py` (예산 1536, batch 10).
준비: `scripts/stage1b_prep.py` → `logs/stage1b/<task>/prompts.jsonl`. 채점: `tar_rlvr/rewards/cova_tasks.py` (+ `tests/test_cova_tasks.py`).

## GT 첫 이미지 확인 (train 전체, 픽셀 기반 — `tar_rlvr/vis/gt_checks.py`)

| | 행 | 다른 문제의 그림 | state-only | leaky |
|---|---|---|---|---|
| maze | 4,320 | 120 (행 4200–4319: 질문 미로와 1칸 정도 다름) | 4,200 (벽·S·E 만, 경로 표시 없음, 격자가 질문과 칸 단위로 일치) | 0 |
| clutrr | 2,688 | 436 mismatch + 220 uncertain (이름 상자가 질문의 이름이 아님) | 1,907 (빨간 관계 라벨 정확히 1개 = 이야기의 첫 관계) | 125 (관계 2–3개, 질의 쌍 직접 연결 등) |

- **clutrr 데이터 버그**: 약 16–24% 행의 그림이 다른 문제의 가계도다 (예: 행 1658 은 Andrew/Laura/James 문제인데 그림 0·1·4 는 Michael/Joseph/Edward).
  trace 텍스트는 맞는 그림을 묘사하므로 텍스트로는 못 잡는다. OCR 이 없어 이름은 Pillow 글꼴 템플릿 매칭 (닫힌 어휘) 으로 읽었고,
  손으로 확인한 24 행에서 오분류 0 (경계 사례는 uncertain → 표본 제외).
- 표본: maze 200 (풀 수 있음 109 / UNSOLVABLE 91, leaky 0), clutrr 200 (leaky 7), prosqa 100 (스캔 전용, 그림 확인 안 함).
- maze 의 S 없는 미로 18 행 (중복 포함) 제외.

## maze — 결과: oracle 효과 없음

**on-demand 는 불가**: greedy 에서 모델이 `<im_start>` 를 한 번도 안 낸다 (0/200) → 개입이 일어나지 않음.
그래서 그림을 completion 첫 토큰에 강제 (`--force-image`, 출력 `logs/stage1b/maze_forced`). Stage 1 math 에서도 그림의 97% 가 토큰 0 이었다.

greedy, 200 프롬프트 (전체 표: `maze_forced_report_greedy.md`):

| 조건 | 전체 | 풀 수 있음 (109) | 풀 수 있는데 UNSOLVABLE 이라 답함 | UNSOLVABLE (91) 에 UNSOLVABLE |
|---|---|---|---|---|
| text_only | 13.5 | **0.0** | 29 | 27 |
| model_drawn | 35.0 | **0.0** | 81 | 70 |
| model_drawn_S0 | 27.0 | **0.0** | 57 | 54 |
| oracle_S2 | 29.0 | **0.0** | 77 | 58 |
| oracle_S0 | 22.0 | **0.0** | 41 | 44 |
| corrupt (S2) | 30.5 | **0.0** | 68 | 61 |
| corrupt_S0 | 23.5 | **0.0** | 46 | 47 |

- 풀 수 있는 미로는 모든 조건에서 0/109. 그림이 있으면 경로 격자를 아예 안 낸다 (text_only 15 개 → 그림 조건 0–3 개).
- 전체 정답률 차이 (oracle_S2 − text_only +15.5pp, p=3e-6) 는 **UNSOLVABLE 쏠림**이다: 모든 조건에서 UNSOLVABLE 답 비율이
  풀 수 있는/없는 미로에서 거의 같다 → 구별 못 함. 그림이 들어가면 UNSOLVABLE 이라고 더 자주 말할 뿐.
- oracle − corrupt: S2 −1.5pp (p=0.76), S0 −1.5pp (p=0.76). 정답 그림이 다른 미로 그림보다 낫지 않다.
- S0 는 S2 보다 나쁘다 (답 미도달 증가: 풀 수 있는 미로 67/109).
- sampled (×8) 는 결론이 바뀔 여지가 없어 돌리지 않았다 (2026-11-26 결정).

채점 주의: maze on-demand greedy (`logs/stage1b/maze`) 는 `\boxed{}` 우선 규칙 수정 전 채점이라 text_only 가 15.0 (수정 후 13.5) 으로 기록돼 있다.
그림이 0 회라 다른 조건은 전부 같은 텍스트다. maze_forced 는 수정 후 채점.

## clutrr — 기본 passrate 스캔 (완료, `clutrr_scan_report.md`)

`oracle_eval.py --kinds sampled --conds text_only,model_drawn` (×8, T=0.8; model_drawn = sketch 지시문 + 그림 뒤 bridge = scan_passrate.py 규칙).
non-leaky 193 문제:

| | pass@1 [95% CI] | pass@8 [95% CI] |
|---|---|---|
| text_only | 0.113 [0.091, 0.138] | 0.451 [0.383, 0.523] |
| model_drawn | 0.106 [0.083, 0.131] | 0.435 [0.363, 0.508] |
| 찍기 기준선 | 무작위 0.056 / 항상 최빈값 (nephew) 0.125 | 무작위 8번 ≈ 0.37 |

- pass@8 기준 (0.7 초과 / 0 근처) 으로는 통과지만 **pass@1 은 찍기 수준**: clutrr 은 주관식이지만 답이 관계어 18 개뿐이다.
- maze 와 달리 그림을 스스로 시작한다 (샘플 53%, 169/193 문제) → on-demand oracle 가능. 스스로 그린 그림은 효과 없음 (그림 샘플 0.107 vs 아닌 샘플 0.106).

## clutrr — oracle_S0: 효과 없음 — `clutrr_oracle_report.md`

조건 text_only / model_drawn / oracle_S0 / corrupt_S0, 예산 1536, on-demand (모델이 `<im_start>` 를 낸 자리에 GT 첫 그림 S0 = 729 토큰).
main = state-only 193 (이름 검사 match + 첫 관계 하나). corrupt 그림도 state-only 193 안에서만 짝 (`prompts.before_corrupt_remap.jsonl` 은 원본).
greedy 에서 그림을 시작한 문제 128/193.

| greedy | text_only | model_drawn | oracle_S0 | corrupt_S0 |
|---|---|---|---|---|
| main (193) | 13.5 | 14.5 | **9.8** | 13.0 |
| 그림 시작 (128) | 15.6 | 13.3 | **6.2** | 10.9 |

- oracle_S0 − corrupt_S0 −3.1pp (3 / 9, McNemar p=0.15), oracle_S0 − text_only −3.6pp (p=0.31). 정답 그림이 틀린 그림보다 낫지 않다 (점추정은 반대 방향).
- S0 주입 뒤 잘림 36% (oracle) / 34% (corrupt) 로 같다. 답을 낸 경우만 봐도 oracle 0.11 vs corrupt 0.18.
- 전 조건이 찍기 기준선 (최빈값 0.125) 근처.

sampled ×8 (스캔의 prefix 재사용 → 그림 세 조건은 샘플 단위로 정확히 짝), 프롬프트 단위 paired bootstrap:

| main (193) | pass@1 | pass@8 |
|---|---|---|
| text_only | 11.3 | 45.1 |
| model_drawn | 10.6 | 43.5 |
| oracle_S0 | 8.3 | 34.7 |
| corrupt_S0 | 8.4 | 36.3 |

- **oracle_S0 − corrupt_S0: pass@1 −0.1pp [−1.2, +1.0], pass@8 −1.6pp [−5.7, +2.6].** 샘플 단위 (그림 낸 815 샘플) −0.2pp [−2.2, +1.8]. 정답 그림과 다른 문제 그림이 구별되지 않는다.
- oracle_S0 − text_only: pass@1 −3.0pp [−5.1, −0.8], pass@8 −10.4pp [−18.1, −2.6]. S0 그림을 넣는 것 자체가 손해 (corrupt 도 같은 크기로 손해).
- 그림 위치: 847 샘플 중 824 가 토큰 0 — "on demand" 는 사실상 "추론 전 그림".

## prosqa — 기본 passrate 스캔 (완료, `prosqa_scan_report.md`)

사실상 2지선다 ("Is Tom a worpus or quimpus?") → pass@8 은 찍기만으로 1 에 가깝다. 100 문제 × 8:

| | pass@1 [95% CI] | pass@8 | 답 도달 | 답했을 때 정답률 |
|---|---|---|---|---|
| text_only | 0.259 [0.219, 0.304] | 0.790 | 55% | 0.47 |
| model_drawn | 0.255 [0.212, 0.301] | 0.760 | 56% | 0.45 |

- 답했을 때 정답률이 동전 던지기 (0.5) 수준, 45% 는 답을 못 낸다 (예산 소진 또는 두 보기를 다 씀). 그림 시작 9% (72/800).
