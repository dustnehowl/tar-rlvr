#!/bin/bash
# Stage 1b TPU 큐 (v2). maze on-demand greedy 는 <im_start> 0/200 → 개입이 안 일어남 → 그림 강제 (--force-image) 로 대체.
# 1) 스모크 2) maze_forced greedy 3) clutrr·prosqa 기본 passrate 스캔 (sampled ×8, text_only + model_drawn) 4) maze_forced sampled
cd /mnt/work/tar-rlvr
PY=/mnt/work/venv311/bin/python
MAZE_CONDS=text_only,model_drawn,model_drawn_S0,oracle_S2,oracle_S0,corrupt,corrupt_S0
run() { while [ ! -f "$2" ]; do sleep 30; done; echo "[queue1b] $(date -u +%FT%TZ) start $*"; $PY -u scripts/oracle_eval.py --batch 10 "$@" || { echo "[queue1b] FAILED $*"; exit 1; }; echo "[queue1b] $(date -u +%FT%TZ) done $*"; }
run --prompts logs/stage1b/maze/prompts.jsonl   --out logs/stage1b/smoke_forced --kinds greedy --limit 3 --force-image --conds model_drawn,model_drawn_S0,oracle_S2,oracle_S0,corrupt_S0
run --prompts logs/stage1b/maze/prompts.jsonl   --out logs/stage1b/maze_forced  --kinds greedy  --force-image --conds $MAZE_CONDS
run --prompts logs/stage1b/clutrr/prompts.jsonl --out logs/stage1b/clutrr       --kinds sampled --conds text_only,model_drawn
run --prompts logs/stage1b/prosqa/prompts.jsonl --out logs/stage1b/prosqa       --kinds sampled --conds text_only,model_drawn
run --prompts logs/stage1b/maze/prompts.jsonl   --out logs/stage1b/maze_forced  --kinds sampled --force-image --conds $MAZE_CONDS
echo "[queue1b] all done"
