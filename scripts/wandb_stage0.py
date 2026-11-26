"""Stage 0 결과(reports/stage0)를 W&B 에 올린다: 문제별 표 [step0 그림 | step150 그림 | 정답 그림 | 정답여부 | 문제 | completion] + contact sheet.

사용: /mnt/work/wandb-venv/bin/python scripts/wandb_stage0.py --entity dustnehowl
"""
import argparse, glob, json, os, re
import wandb

IMG_TOK = re.compile(r"<im_start><S\d>(?:<I\d+>)+<im_end>")


def strip_images(text):
    return IMG_TOK.sub("[IMAGE]", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/stage0")
    ap.add_argument("--run", default="logs/runs/r06-sketch-bridge")
    ap.add_argument("--a", type=int, default=0)
    ap.add_argument("--b", type=int, default=150)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", default="tar-rlvr")
    ap.add_argument("--name", default="stage0-r06-sketches")
    a = ap.parse_args()

    meta = json.load(open(f"{a.dir}/meta.json"))
    ev = {}
    for s in (a.a, a.b):
        ev[s] = {r["qid"]: r for r in map(json.loads, open(f"{a.run}/eval/step_{s:06d}.jsonl")) if r["kind"] == "greedy"}

    run = wandb.init(entity=a.entity, project=a.project, name=a.name, id=a.name, resume="allow", job_type="analysis",
                     config={"stage": 0, "source_run": "r06-sketch-bridge", "steps": [a.a, a.b], "n_prompts": len(meta),
                             "decoder": "TA-Tok AR dtok 256px + LlamaGen VQ, cfg 4.0, seed 0"})
    cols = ["i", "qid", "task", "gold", f"gen_step{a.a}", f"gen_step{a.b}", "gt_cova",
            f"correct_step{a.a}", f"correct_step{a.b}", "question", f"completion_step{a.a}", f"completion_step{a.b}"]
    t = wandb.Table(columns=cols)
    for m in meta:
        p = f"{a.dir}/img/{m['i']:02d}_{m['qid']}"
        gt = f"{p}_gt.png"
        t.add_data(m["i"], m["qid"], m["task"], m["gold"],
                   wandb.Image(f"{p}_step{a.a}.png"), wandb.Image(f"{p}_step{a.b}.png"),
                   wandb.Image(gt) if os.path.exists(gt) else None,
                   m[f"correct_step{a.a}"], m[f"correct_step{a.b}"], m["question"],
                   strip_images(ev[a.a][m["qid"]]["completion"])[:3000], strip_images(ev[a.b][m["qid"]]["completion"])[:3000])
    log = {"stage0/table": t}
    for f in sorted(glob.glob(f"{a.dir}/contact_*.png")):
        log[f"stage0/{os.path.basename(f)[:-4]}"] = wandb.Image(f, caption="step0 | step150 | CoVA GT | problem")
    run.log(log)
    readme = f"{a.dir}/README.md"
    if os.path.exists(readme):
        run.summary["stage0/judgement"] = "(a) relevant 4/30, (b) generic 16/30, (c) garbage 10/30; labeled 0/30; no step0→150 improvement"
        art = wandb.Artifact("stage0-report", type="report")
        art.add_file(readme); art.add_file(f"{a.dir}/meta.json")
        run.log_artifact(art)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
