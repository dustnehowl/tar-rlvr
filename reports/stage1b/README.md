# Stage 1b — 비수학 태스크 oracle (maze 완료, clutrr 대기)

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

## clutrr / prosqa — 기본 passrate 스캔 (진행 중)

이 두 태스크는 기존 passrate 스캔이 없었다 (`data/passrate*.json` 은 math 만).
스캔 = `oracle_eval.py --kinds sampled --conds text_only,model_drawn` (×8, T=0.8; model_drawn = sketch 지시문 + 그림 뒤 bridge 이어쓰기 = scan_passrate.py 와 같은 규칙).
pass@8 이 0.7 초과거나 0 근처면 clutrr oracle_S0 대신 다른 태스크를 고른다. 아니면 clutrr 은 S0 만 (`oracle_S0`, 대조군 `corrupt_S0`) — S2 는 이름 라벨이 깨진다 (Stage 1 round-trip).
