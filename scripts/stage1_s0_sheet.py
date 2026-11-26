"""Stage 1: 모델이 S0 (729 토큰, 강제) 로 그린 그림 vs 스스로 고른 스케일 (보통 S2) 로 그린 그림 vs GT — contact sheet.

같은 프롬프트·같은 prefix (greedy) 에서 model_drawn 과 model_drawn_S0 의 첫 그림을 디코딩한다.
사용: /mnt/work/venv311/bin/python scripts/stage1_s0_sheet.py --run logs/stage1/b1536 --n 20 --out reports/stage1_model_s0.png
"""
import argparse, io, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont
from tar_rlvr.vis.tatok import TATok, parse_blocks

CELL = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/stage1/b1536")
    ap.add_argument("--prompts", default="logs/stage1/prompts.jsonl")
    ap.add_argument("--kind", default="greedy")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/stage1_model_s0.png")
    a = ap.parse_args()
    P = {}
    for l in open(a.prompts):
        r = json.loads(l); P.setdefault(r["qid"], r)
    R = [json.loads(l) for l in open(f"{a.run}/records.jsonl")]
    by = {(r["cond"], r["qid"]): r for r in R if r["kind"] == a.kind and r["sample"] == 0}
    cand = sorted(q for (c, q), r in by.items() if c == "model_drawn_S0" and r["emitted"]
                  and parse_blocks(r["completion"]) and ("model_drawn", q) in by and parse_blocks(by[("model_drawn", q)]["completion"]))
    pick = random.Random(a.seed).sample(cand, min(a.n, len(cand)))
    print(f"[s0sheet] candidates {len(cand)}, picked {len(pick)}")
    from datasets import load_dataset
    import hashlib
    gts = {}
    for t in ("math_vr", "mathnet"):
        for r in load_dataset("tsunghanwu/cova", t, split="train"):
            q = hashlib.md5(r["question"].encode()).hexdigest()[:12]
            if q in pick and q not in gts:
                gts[q] = Image.open(io.BytesIO(r["images"][0])).convert("RGB")
    tk = TATok()
    font = ImageFont.load_default(size=14)
    os.makedirs(os.path.splitext(a.out)[0], exist_ok=True)
    rows, meta = [], []
    for i, q in enumerate(pick):
        cells, info = [], []
        for c in ("model_drawn", "model_drawn_S0"):
            s, codes = parse_blocks(by[(c, q)]["completion"])[0]
            im = tk.decode(s, codes)
            im.save(f"{os.path.splitext(a.out)[0]}/{i:02d}_{q}_{c}.png")
            cells.append(im.resize((CELL, CELL)))
            info.append(f"{c}: S{s}/{len(codes)} correct={int(by[(c, q)]['correct'])}")
        cells.append(gts[q].resize((CELL, CELL)))
        row = Image.new("RGB", (CELL * 3 + 4 + 520, CELL + 4), "white")
        for k, im in enumerate(cells):
            row.paste(im, (k * (CELL + 2), 2))
        d = ImageDraw.Draw(row)
        d.multiline_text((CELL * 3 + 12, 6), f"#{i} {q}\n" + "\n".join(info) + "\n\n" + "\n".join(
            P[q]["question"].replace("\n", " ")[k:k + 70] for k in range(0, 700, 70)), fill="black", font=font)
        rows.append(row)
        meta.append({"i": i, "qid": q, "info": info})
        print(f"[s0sheet] {i + 1}/{len(pick)} {q}", flush=True)
    head = Image.new("RGB", (rows[0].width, 24), "white")
    ImageDraw.Draw(head).text((6, 4), "model_drawn (own scale) | model_drawn_S0 (forced <S0>, 729 tokens) | CoVA GT | problem",
                              fill="black", font=font)
    sheet = Image.new("RGB", (rows[0].width, 24 + sum(r.height for r in rows)), "white")
    sheet.paste(head, (0, 0)); y = 24
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    sheet.save(a.out)
    json.dump(meta, open(os.path.splitext(a.out)[0] + "/meta.json", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
