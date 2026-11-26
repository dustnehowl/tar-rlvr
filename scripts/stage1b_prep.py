"""Stage 1b 준비 (CPU): maze / clutrr / prosqa 프롬프트 + GT 첫 이미지 확인 + TA-Tok 토큰 → logs/stage1b/<task>/prompts.jsonl

Stage 1 (math) 과 같은 형식의 prompts.jsonl 을 만든다 → scripts/oracle_eval.py 를 그대로 쓴다.

GT 첫 이미지 확인 (tar_rlvr/vis/gt_checks.py, 픽셀 기반, train 전체에 대해 실행하고 통계를 남김)
  maze  : 첫 그림의 격자를 읽어 문제 미로와 칸 단위로 같고, 경로·탐색 표시 색이 없으면 state-only.
  clutrr: 이름 상자가 질문의 이름과 맞는지 (match / mismatch / uncertain), 빨간 라벨(새 관계) 정확히 1개,
          회색 라벨 0개, 두 질의 인물이 서로 이어지지 않았으면 state-only (= 이야기의 첫 관계 하나만).
  - 그림이 다른 문제의 것 (maze 격자 불일치, clutrr 이름 mismatch/uncertain) → 표본에서 **제외** (GT 가 아님).
  - 같은 문제의 그림이지만 state-only 가 아님 → 표본에 남기고 leaky=True 로 표시 (보고서 main 에서 제외).
  prosqa: 스캔 전용 (oracle 안 함) → 그림 확인·인코딩 안 함.

표본: train split, 태스크마다 --n 개 (seed 0). maze 는 S/E 가 정확히 하나씩인 미로만 (CoVA 에 S 없는 행 5개).
토큰: GT 첫 이미지 S2 (81) + S0 (729). corrupt = 같은 태스크 안 고정점 없는 순열 (seed 0) 의 GT 첫 이미지.

사용: /mnt/work/venv311/bin/python scripts/stage1b_prep.py --tasks maze,clutrr,prosqa
"""
import argparse, collections, hashlib, io, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

REV = "a8080bcb803f2cf76c5debf8c6fc233a165aefe3"
GROUP = {"maze": "game", "clutrr": "graph", "prosqa": "graph"}


def load(task):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("tsunghanwu/cova", f"train_split/{GROUP[task]}/{task}.parquet", repo_type="dataset", revision=REV)
    rows = pq.read_table(p).to_pylist()
    for r in rows:
        r["images"] = [im["bytes"] if isinstance(im, dict) else im for im in r["images"]]
    return rows


def qid(q):
    return hashlib.md5(q.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="maze,clutrr,prosqa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-prosqa", type=int, default=100)
    ap.add_argument("--out", default="logs/stage1b")
    ap.add_argument("--config", default="configs/tpu_v6e_7b.yaml")
    a = ap.parse_args()
    import yaml
    from tar_rlvr.data.synthetic import SYSTEM_SKETCH, SYSTEM_NEUTRAL
    from tar_rlvr.model.tar import load_tokenizer
    from tar_rlvr.train.grpo import format_prompt
    from tar_rlvr.rewards.cova_tasks import maze_from_question, prosqa_ref
    from tar_rlvr.vis import gt_checks
    cfg = yaml.safe_load(open(a.config))
    tok = load_tokenizer(cfg["model"]["tokenizer_path"])
    tk = None

    for task in a.tasks.split(","):
        rows = load(task)
        os.makedirs(f"{a.out}/{task}", exist_ok=True)
        # ---------------------------------------------------- 전체 train 에 대해 첫 그림 확인
        t0 = time.time()
        checks, stats = [], collections.Counter()
        pool = sorted({n for r in rows for n in re.findall(r"\[(\w+)\]", r["question"])}) if task == "clutrr" else None
        for i, r in enumerate(rows):
            c = dict(idx=i, qid=qid(r["question"]))
            if task == "maze":
                raw = maze_from_question(r["question"])
                c["raw_maze"] = raw
                c["valid_maze"] = raw.count("S") == 1 and raw.count("E") == 1
                so, why = gt_checks.maze_check(r["images"][0], raw)
                c.update(own_image=why != "grid_differs_from_question" and not why.startswith("unknown"),
                         state_only=so, reason=why)
                stats[f"valid_maze={c['valid_maze']}"] += 1
            elif task == "clutrr":
                qn = set(re.findall(r"\[(\w+)\]", r["question"]))
                meta = json.loads(r["metadata"])
                cc = gt_checks.clutrr_check(r["images"][0], qn, pool, tuple(meta.get("query_people") or ()))
                c.update(own_image=cc["names"] == "match", names=cc["names"], margin=round(cc["margin"], 3),
                         n_red_labels=cc["n_red_labels"], n_gray_labels=cc["n_gray_labels"],
                         query_pair_edge=bool(cc["query_pair_edge"]), state_only=cc["state_only"],
                         reason=cc["reason"], reads=cc["reads"])
                stats[f"names={cc['names']}"] += 1
                stats[f"red_labels={min(cc['n_red_labels'], 3)}"] += 1
            else:
                c.update(own_image=True, state_only=None, reason="not_checked")
            if task != "prosqa":
                stats[f"own_image={c['own_image']}"] += 1
                stats[f"state_only={c['state_only']}"] += 1
                if c["own_image"]:
                    stats[f"own_image&state_only={c['state_only']}"] += 1
            checks.append(c)
            if (i + 1) % 500 == 0:
                print(f"[prep1b] {task} check {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        with open(f"{a.out}/{task}/gt_checks_train.jsonl", "w") as f:
            for c in checks:
                f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
        json.dump(dict(stats), open(f"{a.out}/{task}/gt_checks_summary.json", "w"), indent=1)
        print(f"[prep1b] {task} checks: {dict(stats)}", flush=True)

        # ---------------------------------------------------- 표본
        seen, cand = set(), []
        for r, c in zip(rows, checks):
            if c["qid"] in seen:
                continue
            seen.add(c["qid"])
            if task == "maze" and not c["valid_maze"]:
                continue
            if not c["own_image"]:
                continue
            cand.append((r, c))
        n = a.n_prosqa if task == "prosqa" else a.n
        pick = random.Random(0).sample(cand, n)
        print(f"[prep1b] {task}: candidates {len(cand)} (unique qid, own image) → sample {len(pick)}", flush=True)

        out = []
        for r, c in pick:
            if task == "maze":
                ref = {"answer": r["answer"].strip(), "raw_maze": c["raw_maze"]}
                extra = dict(solvable=r["answer"].strip() != "UNSOLVABLE", grid=f"{len(c['raw_maze'].split()[0])}x{len(c['raw_maze'].split())}")
            elif task == "clutrr":
                ref = {"answer": r["answer"].strip()}
                extra = dict(relation=r["answer"].strip(), n_hops=len(re.findall(r"\.\s", r["question"])))
            else:
                ref = prosqa_ref(r["question"], r["answer"])
                extra = {}
            o = {"qid": c["qid"], "split": "train", "subset": task, "task": task, "ref": json.dumps(ref),
                 "is_mc": False, "question": r["question"],
                 "prompt_sketch": format_prompt(tok, f"{SYSTEM_SKETCH}\n{r['question']}"),
                 "prompt_neutral": format_prompt(tok, f"{SYSTEM_NEUTRAL}\n{r['question']}"),
                 "n_gt_images": len(r["images"]),
                 "leaky": (not c["state_only"]) if c["state_only"] is not None else False,
                 "leaky_reason": c["reason"] if c["state_only"] is False else "", **extra}
            if task != "prosqa":
                if tk is None:
                    from tar_rlvr.vis.tatok import TATok, to_token_string
                    tk = TATok()
                img = Image.open(io.BytesIO(r["images"][0])).convert("RGB")
                o["gt_tokens"] = to_token_string(*tk.encode(img, 2))
                o["gt_tokens_s0"] = to_token_string(*tk.encode(img, 0))
            out.append(o)
            print(f"[prep1b] {task} {len(out)}/{len(pick)} {o['qid']} leaky={o['leaky']} {o['leaky_reason']}", flush=True)
        if task != "prosqa":
            perm = list(range(len(out)))
            rng = random.Random(0)
            while True:
                rng.shuffle(perm)
                if all(i != p for i, p in enumerate(perm)):
                    break
            for i, p in enumerate(perm):
                out[i]["corrupt_qid"] = out[p]["qid"]
                out[i]["corrupt_tokens"] = out[p]["gt_tokens"]
                out[i]["corrupt_tokens_s0"] = out[p]["gt_tokens_s0"]
        path = f"{a.out}/{task}/prompts.jsonl"
        with open(path + ".tmp", "w") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
        print(f"[prep1b] {task}: wrote {len(out)} → {path}; leaky {sum(o['leaky'] for o in out)}", flush=True)


if __name__ == "__main__":
    main()
