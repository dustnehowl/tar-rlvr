#!/bin/bash
# 큐 v4 (2026-11-26): clutrr 스캔 (실행 중, pid 33637) → probe 스모크 → probe 본실행 → prosqa 스캔
cd /mnt/work/tar-rlvr
PY=/mnt/work/venv311/bin/python
while kill -0 33637 2>/dev/null; do sleep 30; done
echo "[queue1b] $(date -u +%FT%TZ) clutrr scan process exited"
step() { echo "[queue1b] $(date -u +%FT%TZ) start $1"; shift; "$@" && echo "[queue1b] $(date -u +%FT%TZ) done" || { echo "[queue1b] FAILED"; return 1; }; }
step "probe smoke" $PY -u scripts/probe_continuation.py --limit 2 --G 2 --out logs/probe_continuation_smoke \
  && step "probe full" $PY -u scripts/probe_continuation.py --out logs/probe_continuation
step "prosqa scan" $PY -u scripts/oracle_eval.py --batch 10 --prompts logs/stage1b/prosqa/prompts.jsonl --out logs/stage1b/prosqa --kinds sampled --conds text_only,model_drawn
echo "[queue1b] all done"
