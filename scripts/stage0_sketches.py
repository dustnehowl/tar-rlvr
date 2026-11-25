"""Stage 0: r06 이 그리는 그림 vs CoVA 정답 그림. 학습 없음, CPU.

held-out eval 의 greedy completion 중 step A(기본 0)와 step B(기본 150) 둘 다 완결된 그림이 있는 문제에서
n 개를 골라, 행마다 [step A 생성 | step B 생성 | CoVA 첫 이미지 | 문제 텍스트] 로 contact sheet 를 만든다.

사용: python scripts/stage0_sketches.py --run logs/runs/r06-sketch-bridge --n 15 --out reports/stage0
"""
import argparse, hashlib, io, json, os, random, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont
from tar_rlvr.vis.tatok import TATok, parse_blocks

CELL = 256
TEXT_W = 560


def cova_rows(tasks=("math_vr", "mathnet")):
    from datasets import load_dataset
    out = {}
    for t in tasks:
        for r in load_dataset("tsunghanwu/cova", t, split="train"):
            q = hashlib.md5(r["question"].encode()).hexdigest()[:12]
            out.setdefault(q, {"task": t, "question": r["question"], "answer": r["answer"], "images": r["images"],
                               "trace": r["interleaved_reasoning"]})
    return out


def load_eval(run, step):
    rows = [json.loads(l) for l in open(f"{run}/eval/step_{step:06d}.jsonl")]
    return {r["qid"]: r for r in rows if r["kind"] == "greedy"}


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/runs/r06-sketch-bridge")
    ap.add_argument("--a", type=int, default=0)
    ap.add_argument("--b", type=int, default=150)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--per-sheet", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/stage0")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/img", exist_ok=True)

    ea, eb = load_eval(a.run, a.a), load_eval(a.run, a.b)
    both = sorted(q for q in ea if q in eb and parse_blocks(ea[q]["completion"]) and parse_blocks(eb[q]["completion"]))
    pick = random.Random(a.seed).sample(both, min(a.n, len(both)))
    print(f"[stage0] 두 시점 모두 그림 있는 문제 {len(both)}개 중 {len(pick)}개")
    cova = cova_rows()
    tk = TATok()
    f_small, f_head = font(13), font(15)
    rows, meta = [], []
    for i, q in enumerate(pick):
        cells = []
        for tag, ev in (("A", ea), ("B", eb)):
            s, codes = parse_blocks(ev[q]["completion"])[0]
            im = tk.decode(s, codes)
            im.save(f"{a.out}/img/{i:02d}_{q}_step{a.a if tag == 'A' else a.b}.png")
            cells.append(im.resize((CELL, CELL)))
        c = cova.get(q)
        if c and c["images"]:
            gt = Image.open(io.BytesIO(c["images"][0])).convert("RGB")
            gt.save(f"{a.out}/img/{i:02d}_{q}_gt.png")
            cells.append(gt.resize((CELL, CELL)))
        else:
            cells.append(Image.new("RGB", (CELL, CELL), "lightgray"))
        txt = Image.new("RGB", (TEXT_W, CELL), "white")
        d = ImageDraw.Draw(txt)
        qtext = (c["question"] if c else "?").replace("# Question", "").replace("Problem:", "").strip()
        d.text((6, 4), f"#{i} {q} ({c['task'] if c else '?'})  gold={c['answer'][:30] if c else '?'}  "
                       f"correct: step{a.a}={int(ea[q]['correct'])} step{a.b}={int(eb[q]['correct'])}", fill="black", font=f_head)
        d.multiline_text((6, 26), "\n".join(textwrap.wrap(qtext, 78)[:13]), fill="black", font=f_small)
        row = Image.new("RGB", (CELL * 3 + TEXT_W + 8, CELL + 4), "white")
        for k, im in enumerate(cells):
            row.paste(im, (k * (CELL + 2), 2))
        row.paste(txt, (CELL * 3 + 8, 2))
        rows.append(row)
        meta.append({"i": i, "qid": q, "task": c["task"] if c else None, "gold": c["answer"] if c else None,
                     f"correct_step{a.a}": ea[q]["correct"], f"correct_step{a.b}": eb[q]["correct"],
                     "question": qtext[:500]})
        print(f"[stage0] {i + 1}/{len(pick)} {q}", flush=True)
    head = Image.new("RGB", (rows[0].width, 26), "white")
    ImageDraw.Draw(head).text((6, 4), f"generated @ step {a.a}  |  generated @ step {a.b}  |  CoVA first image (GT)  |  problem", fill="black", font=f_head)
    for s in range(0, len(rows), a.per_sheet):
        part = rows[s:s + a.per_sheet]
        sheet = Image.new("RGB", (rows[0].width, 26 + sum(r.height for r in part)), "white")
        sheet.paste(head, (0, 0)); y = 26
        for r in part:
            sheet.paste(r, (0, y)); y += r.height
        p = f"{a.out}/contact_{s // a.per_sheet:02d}.png"; sheet.save(p); print("saved", p)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
