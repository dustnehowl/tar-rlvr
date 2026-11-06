#!/bin/bash
# TPU VM 첫 부팅 시. 가이드의 Persistent Disk(/mnt/data) 전제.
set -e
sudo apt update && sudo apt install -y git
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd /mnt/data
[ -d tar-rlvr ] || git clone https://github.com/dustnehowl/tar-rlvr.git
cd tar-rlvr
export UV_CACHE_DIR=/mnt/data/uv-cache   # VM 재생성 시 재다운로드 방지
export HF_HOME=/mnt/data/hf
uv sync --extra tpu --extra dev
uv run python scripts/check_env.py
uv run pytest -q
uv run python scripts/train.py --config configs/smoke_cpu.yaml --max-steps 2
