"""학습 로그에서 창(window) 단위 추이 추출: 정답률, 시각 사용률, 시각 샘플 정답률, 재그림률.
사용: python scripts/summarize_log.py logs/run1.log [window=20] [--rescored logs/run1.rescored.jsonl [--field r_new]]
  --rescored : scripts/rescore_log.py 출력으로 reward 를 바꿔 끼운다 (기본 필드 r_new = 현재 verifier 기준)"""
import json, re, sys

args = sys.argv[1:]
rescored = field = None
if "--rescored" in args:
    i = args.index("--rescored"); rescored = args[i + 1]; del args[i:i + 2]
if "--field" in args:
    i = args.index("--field"); field = args[i + 1]; del args[i:i + 2]
path = args[0]
win = int(args[1]) if len(args) > 1 else 20
txt = open(path, encoding="utf-8", errors="replace").read()

# [3→4] 블록마다 "샘플 k: reward=… adv=… …tail" 줄을 모은다
blocks = re.split(r"\[3→4\. reward 합산", txt)[1:]
rows = []
for b in blocks:
    for m in re.finditer(r"샘플 \d+: reward=([+-]\d+\.\d+)\s+adv=[+-]\d+\.\d+\s+…(.*)", b):
        r, tail = float(m.group(1)), m.group(2)
        rows.append((r, "[IMG" in tail or "<im_start>" in tail))
# 롤아웃 블록의 completion 전문에서 시각 여부를 더 정확히 셈 (tail 100자엔 그림이 안 보일 수 있음)
comp_blocks = re.split(r"\[2\. 롤아웃\] step", txt)[1:]
vis_flags = []
for b in comp_blocks:
    for c in re.split(r"\n#\d+\n--- completion\n", b)[1:]:
        c = c.split("\n====")[0]
        vis_flags.append("[IMG" in c)
n = min(len(rows), len(vis_flags))
rows = [(rows[i][0], vis_flags[i]) for i in range(n)]
if rescored:
    rs = [json.loads(l)[field or "r_new"] for l in open(rescored, encoding="utf-8")]
    if len(rs) != len(rows):
        sys.exit(f"--rescored 행 수 {len(rs)} != 로그 샘플 수 {len(rows)} — 같은 로그로 만든 파일인지 확인")
    rows = [(r, v) for r, (_, v) in zip(rs, rows)]
    print(f"[reward = {rescored} 의 {field or 'r_new'}]")

print(f"{'steps':>10} {'acc':>6} {'vis%':>6} {'acc_vis':>8} {'acc_txt':>8}  n")
G = 4
import re as _re
m = _re.search(r"G=(\d+)", txt); G = int(m.group(1)) if m else 4
per_step = G * 2
for s in range(0, len(rows), win * per_step):
    chunk = rows[s:s + win * per_step]
    if not chunk: break
    acc = sum(r >= 1.0 for r, _ in chunk) / len(chunk)
    vis = [r for r, v in chunk if v]; txt_ = [r for r, v in chunk if not v]
    acc_v = sum(r >= 1.0 for r in vis) / len(vis) if vis else float("nan")
    acc_t = sum(r >= 1.0 for r in txt_) / len(txt_) if txt_ else float("nan")
    a, b_ = s // per_step, min((s + win * per_step) // per_step, len(rows) // per_step)
    print(f"{a:>4}-{b_:<5} {acc:6.3f} {len(vis)/len(chunk):6.2f} {acc_v:8.3f} {acc_t:8.3f}  {len(chunk)}")