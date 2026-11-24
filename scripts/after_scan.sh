#!/bin/bash
# 스캔이 끝나면 passrate_1200 으로 config 갱신 후 학습 시작
cd /mnt/work/tar-rlvr
source /mnt/work/venv311/bin/activate; export HF_HOME=/mnt/work/hf
while pgrep -f "scripts/scan_passrate.py" > /dev/null; do sleep 60; done
if [ ! -s data/passrate_1200.json ]; then echo "scan failed: no passrate_1200.json"; exit 1; fi
sed -i 's|^  passrate_file: .*|  passrate_file: data/passrate_1200.json|' configs/tpu_v6e_7b.yaml
grep -n "passrate_file\|num_generations\|max_steps" configs/tpu_v6e_7b.yaml
python scripts/train.py --config configs/tpu_v6e_7b.yaml --debug 2>&1 | tee logs/tpu_run3.log
