#!/bin/bash
# 큐 v5 (2026-11-26): 큐 v4 (prosqa 스캔) 끝나면 clutrr oracle_S0. 200 프롬프트 (샘플 prefix 는 스캔 것 재사용),
# 보고서 main = state-only 193. greedy 는 4 조건 전부 새로, sampled 는 text_only/model_drawn 이미 있음 → oracle_S0, corrupt_S0 만.
cd /mnt/work/tar-rlvr
while kill -0 38424 2>/dev/null; do sleep 30; done
echo "[queue1b] $(date -u +%FT%TZ) start clutrr oracle_S0"
/mnt/work/venv311/bin/python -u scripts/oracle_eval.py --batch 10 --budget 1536 --prompts logs/stage1b/clutrr/prompts.jsonl --out logs/stage1b/clutrr \
  --kinds greedy,sampled --conds text_only,model_drawn,oracle_S0,corrupt_S0 \
  && echo "[queue1b] $(date -u +%FT%TZ) done clutrr oracle_S0" || echo "[queue1b] FAILED clutrr oracle_S0"
