import argparse, yaml
from tar_rlvr.train.grpo import train

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--max-steps", type=int, default=None)
a = p.parse_args()
with open(a.config) as f:
    cfg = yaml.safe_load(f)
train(cfg, max_steps=a.max_steps)
