"""logs/visual_samples.jsonl 의 <I..> 토큰을 Tar de-tokenizer 로 PNG 복원. CPU 로 몇 장만.
사용: python scripts/decode_visual.py --n 8 --out logs/vis
"""
import argparse, json, re, sys, os
sys.path.insert(0, "/mnt/work/Tar")
import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download

SEQ = {0: 729, 1: 169, 2: 81}   # <S0>/<S1>/<S2> → 토큰 수

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="logs/visual_samples.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="logs/vis")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--any-reward", action="store_true", help="정답/오답 반반 대신 그림 있는 샘플을 앞에서부터")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from tok.mm_autoencoder import MMAutoEncoder
    vt = MMAutoEncoder(
        ar_path=hf_hub_download("ByteDance-Seed/Tar-TA-Tok", "ar_dtok_lp_256px.pth"),
        encoder_path=hf_hub_download("ByteDance-Seed/Tar-TA-Tok", "ta_tok.pth"),
        decoder_path=hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt"),
        encoder_args={"input_type": "rec"}, decoder_args={},
    ).eval().to(dtype=torch.float32, device="cpu")
    torch.set_grad_enabled(False)

    rows = [json.loads(l) for l in open(a.jsonl)]
    rows = [r for r in rows if "<im_end>" in r["completion"]]
    if a.any_reward:
        pick = rows[: a.n]
    else:
        pick = [r for r in rows if r["reward"] >= 1][: a.n // 2] + [r for r in rows if r["reward"] < 1][: a.n - a.n // 2]
    for k, r in enumerate(pick):
        m = re.search(r"<im_start><S(\d)>((?:<I\d+>)+)<im_end>", r["completion"])
        if not m:
            continue
        scale = int(m.group(1)); n = SEQ[scale]
        code = [int(x) for x in re.findall(r"<I(\d+)>", m.group(2))][:n]
        code += [0] * (n - len(code))
        vt.ar_model.cls_token_num = n
        vt.encoder.pool_scale = scale + 1
        img = vt.decode_from_encoder_indices(torch.tensor(code).unsqueeze(0), {"cfg_scale": a.cfg})
        img = Image.fromarray(img[0].numpy())
        q = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", r["prompt"]).split("assistant")[0][-300:]
        canvas = Image.new("RGB", (img.width, img.height + 120), "white"); canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((5, img.height + 5), f"S{scale} reward={r['reward']:.0f}\n" + "\n".join(q[i:i+80] for i in range(0, len(q), 80))[:400], fill="black")
        path = f"{a.out}/{k:02d}_S{scale}_r{int(r['reward'])}.png"; canvas.save(path); print("saved", path)

if __name__ == "__main__":
    main()
