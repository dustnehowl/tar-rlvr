#!/bin/bash
# TPU VM 첫 부팅 시 (startup.sh 에서 호출하거나 수동 실행). 가이드의 Persistent Disk(/mnt/data) 전제.
set -e
sudo apt update && sudo apt install -y python3-pip python3-venv git
cd /mnt/data
[ -d tpu-env ] || python3 -m venv tpu-env
source tpu-env/bin/activate
pip install -U pip
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
[ -d tar-rlvr ] || git clone https://github.com/YOUR_ORG/tar-rlvr.git
cd tar-rlvr && pip install -e ".[tpu,dev]"
export HF_HOME=/mnt/data/hf   # 모델 캐시를 디스크에 (VM 재생성 시 재다운로드 방지)
python scripts/check_env.py
pytest -q
python scripts/train.py --config configs/smoke_cpu.yaml --max-steps 2   # TPU에서도 tiny 스모크 먼저
