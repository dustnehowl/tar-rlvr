#!/bin/bash
# 학습을 순서대로 돌린다. 각 run 마다 W&B 업로더(사이드카)를 붙이고, 학습이 끝나면 다음 run 으로.
# 사용: WANDB_API_KEY=... nohup scripts/run_queue.sh <config>:<run 이름> [<config>:<run 이름> ...] > logs/queue.log 2>&1 &
# 같은 run 이름으로 다시 돌리면 checkpoint_dir/<이름> 에서 이어서 학습한다 (VM 재부팅 후 재개용).
set -u
cd "$(dirname "$0")/.."
source /mnt/work/venv311/bin/activate
export HF_HOME=${HF_HOME:-/mnt/work/hf}
[ -f .env ] && { set -a; . ./.env; set +a; }   # WANDB_API_KEY (git 에 안 올라감)
ENTITY=${WANDB_ENTITY:-dustnehowl}

for item in "$@"; do
  cfg=${item%%:*}; name=${item#*:}
  echo "[queue] $(date -u +%FT%TZ) 시작: $name ($cfg)"
  python scripts/train.py --config "$cfg" --run-name "$name" --debug >> "logs/$name.log" 2>&1 &
  pid=$!
  echo "[queue] train PID $pid → logs/$name.log"
  # 모델 로드 후 텔레메트리 디렉토리가 생기면 업로더 시작 (최대 1시간 대기)
  for _ in $(seq 1 120); do
    [ -f "logs/runs/$name/config.json" ] && break
    kill -0 $pid 2>/dev/null || break
    sleep 30
  done
  if [ -f "logs/runs/$name/config.json" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    /mnt/work/wandb-venv/bin/python scripts/wandb_sync.py "logs/runs/$name" --entity "$ENTITY" --follow --until-pid $pid \
      >> "logs/wandb_sync_$name.log" 2>&1 &
    echo "[queue] wandb_sync PID $!"
  fi
  wait $pid
  echo "[queue] $(date -u +%FT%TZ) 종료: $name (exit $?)"
done
echo "[queue] 전부 끝"
