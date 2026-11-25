# Stage 0 — r06 가 지금 무엇을 그리나 (학습 없음)

- 소스: r06-sketch-bridge held-out eval 의 greedy completion. 같은 문제를 **step 0 (학습 전)** 과 **step 150** 에서 비교.
  두 시점 모두 완결된 그림이 있는 문제 60개 중 15개 무작위 (seed 0) → 생성 그림 30장 + CoVA 정답 첫 이미지 15장.
- 디코딩: `tar_rlvr/vis/tatok.py` (TA-Tok AR 디토크나이저 + LlamaGen VQ, CPU, 장당 ~16 s, cfg 4.0).
  Tar 는 항상 `<im_start><S2>` + `<I…>` 81개 + `<im_end>` 로 그린다 (eval 의 그림 88/88 전부 S2·81토큰).
- 정답 이미지: HF `tsunghanwu/cova` 의 `images` 컬럼 (PNG 바이트, 1024×1024, reasoning 의 첫 `<img>` 자리). 따로 받을 필요 없음.
- 파일: `contact_00.png`–`contact_02.png` (행 = [step 0 | step 150 | GT | 문제]), 개별 PNG `img/`, 문제 목록 `meta.json`.
  재현: `python scripts/stage0_sketches.py --out reports/stage0`.

## 판정 (눈으로, 30장)

| 분류 | 장 수 | 비율 | 예 |
|---|---|---|---|
| (a) 문제를 그림 (도형 종류/구조가 맞음) | 4 | 13% | #6 원 + 육각형 (두 시점), #10 마름모 + 대각선 (두 시점) |
| (b) 일반 도형, 문제와 무관하거나 겨우 같은 계열 | 16 | 53% | 격자 (좌표 문제), 별, 사각형, 원, 세로 막대, 다이아몬드 |
| (c) 쓰레기 (글자 덩어리, 낙서, 빈 화면) | 10 | 33% | "DRR", "WRR", "XEY", "DRAWING TO THE MODEL", 빈 회색 화면 2문제 |

- **라벨이 있는 그림은 0장.** 점 이름 (A, B, C, P, O), 좌표축 눈금, 주어진 길이·각도 같은 문제 정보가 들어간 그림이 하나도 없다. 정답 그림은 전부 라벨이 있는 교과서식 도형이다.
- **"draw" 라는 글자를 그린다.** 생성 그림에 "DRAW", "DRR", "WRR", "DRAWING TO THE MODEL" 같은 글자가 자주 나온다. sketch 지시문 ("You may sketch…") 의 단어가 이미지로 새는 것으로 보인다.
- **step 0 → step 150 개선 없음.** 15문제 중 그림 품질이 나아진 문제는 없다. 같은 문제에서 비슷한 그림을 그리거나 다른 일반 도형으로 바뀐다. RL 이 그림의 내용을 바꾸지 못했다 (보상이 그림 내용에 닿지 않으니 당연).
- **정답 여부와 그림은 무관해 보인다.** 30개 completion 중 정답 7개. 그중 (a) 그림은 2개 (#10 두 시점) 뿐이고, 나머지 5개 (#0 step150, #12 두 시점, #13 step0, #14 step0) 는 (b)/(c) 그림 — 빈 화면을 그리고도 맞힌 경우 2개. (a) 인 #6 은 두 시점 모두 오답.

**결론**: 지금 Tar 가 그리는 그림은 문제 상태를 담고 있지 않다. 생성 그림이 추론에 정보를 더할 수 없으므로, r06 에서 그림 사용이 늘어난 것은 그림 내용 때문이 아니다.
Stage 1 (정답 그림을 넣어주면 도움이 되나) 은 이 결과와 무관하게 의미가 있다. 좋은 그림의 상한을 재는 실험이기 때문.

## Stage 1 준비 중 확인한 것

- 디토크나이저 로드: 체크포인트 pickle 에 `easydict.EasyDict` 가 있어 torch 2.14 의 `weights_only=True` 로는 안 읽힌다.
  `pickletools` 로 참조 전역을 정적으로 확인한 뒤 (텐서 복원·OrderedDict·EasyDict 4개뿐), 그 4개만 허용하는 unpickler 로 읽는다 (`tar_rlvr/vis/safe_load.py`).
- transformers 5.12 의 `Siglip2VisionModel` 은 `.vision_model` 래퍼가 없어 TA-Tok 체크포인트 키와 안 맞는다 → 로드 시 `decoder.vision_model.X` → `decoder.X` 로 맞춘다 (`tatok.py`, `/mnt/work/Tar` 원본은 그대로).
- 인코더 (이미지 → 토큰) 쪽은 아직 실행하지 않았다. Stage 1 첫 작업이 round-trip 검증.
