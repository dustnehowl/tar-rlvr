# tar-rlvr

Tar-7B(Qwen2.5-7B + 65,536 시각 토큰)에 CoVA 태스크로 RLVR(GRPO). JAX/Tunix, TPU v6e.

## 왜 이 구조인가
- Tar는 vocab만 확장된 Qwen2.5 → Tunix qwen2 구현에 `vocab_size`만 덮어쓰면 로드됨 (`tar_rlvr/model/tar.py`).
- CoVA 문제는 텍스트 입력 → RL 중엔 TA-Tok 인코더(PyTorch) 불필요. 시각 토큰은 모델이 생성하는 latent workspace, reward는 `<answer>` 텍스트만 봄.
- Reward는 전부 순수 파이썬 verifier → **하드웨어 없이 100% 테스트** (`tests/`).
- `configs/smoke_cpu.yaml`: 랜덤 tiny 모델 + 합성 데이터로 GRPO 루프 end-to-end. CI에서 돌아감.

## 검증 순서 (돈 쓰기 전에)
1. `pip install -e ".[cpu,dev]" && pytest -q` — verifier
2. `python scripts/check_env.py` — Tunix API 심볼 존재 확인 (API 드리프트 감지)
3. `python scripts/train.py --config configs/smoke_cpu.yaml --max-steps 2` — 루프
4. `python scripts/inspect_cova.py` → `configs/tpu_v6e_7b.yaml` 의 컬럼/`TASK_MAP` 채우기
5. TPU VM: `scripts/tpu_setup.sh` → 같은 스모크 → `configs/tpu_v6e_7b.yaml`

## 구조
```
configs/          smoke_cpu.yaml, tpu_v6e_7b.yaml
tar_rlvr/model/   tar.py(로드/LoRA), tiny_tokenizer.py
tar_rlvr/data/    synthetic.py, cova.py
tar_rlvr/rewards/ parsing.py, verifiers.py(task별), __init__.py(Tunix reward_fns)
tar_rlvr/train/   grpo.py (Tunix 어댑터 — API 바뀌면 여기만)
scripts/          train.py, check_env.py, inspect_cova.py, tpu_setup.sh
tests/            test_rewards.py
```

## TODO
- [ ] CoVA 스키마 확인 → `cova.py` TASK_MAP / ref 구성 (sudoku: puzzle, coloring: edges,k)
- [ ] Tar 토크나이저에 `<I*>` added_tokens 있는지 확인 (`load_tokenizer` assert)
- [ ] Tunix 설치 버전에 맞춰 `grpo.py` 어댑터 필드명 맞추기
- [ ] 롤아웃 엔진 vanilla → vLLM-TPU 전환 (7B 샘플링 속도)
- [ ] 평가: CoVA-Bench(1,700) 스크립트
