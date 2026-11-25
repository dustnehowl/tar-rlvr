"""Tar 시각 토큰 <-> 이미지. TA-Tok 인코더 + AR 디토크나이저(LlamaGen VQ) — PyTorch, /mnt/work/Tar 코드 사용.

Tar 는 `<im_start><S2>` + `<I…>` 81개 + `<im_end>` 로 그림을 그린다 (S0=729, S1=169, S2=81 토큰).
"""
from __future__ import annotations
import re, sys
import numpy as np

SEQ = {0: 729, 1: 169, 2: 81}
_BLOCK = re.compile(r"<im_start><S(\d)>((?:<I\d+>)+)<im_end>")


def parse_blocks(text: str):
    """[(scale, [code…])…] — 완결된 그림 블록만."""
    return [(int(m.group(1)), [int(x) for x in re.findall(r"<I(\d+)>", m.group(2))]) for m in _BLOCK.finditer(text)]


def to_token_string(scale: int, codes) -> str:
    return f"<im_start><S{scale}>" + "".join(f"<I{int(c)}>" for c in codes) + "<im_end>"


def _patch_ta_tok_keys():
    """transformers 5.x 의 Siglip2VisionModel 은 `.vision_model` 래퍼가 없다 → 체크포인트 키
    `decoder.vision_model.X` 를 `decoder.X` 로 맞춘다 (모델에 없는 키에만 적용, 나머지는 strict 그대로)."""
    import torch
    from tok.ta_tok import TextAlignedTokenizer
    if getattr(TextAlignedTokenizer, "_tar_rlvr_patched", False):
        return

    @classmethod
    def from_checkpoint(cls, ckpt, load_teacher=True, **kwargs):
        ckpt = torch.load(ckpt, map_location="cpu")
        model = cls(**kwargs, **ckpt["model"]["args"])
        sd = ckpt["model"]["sd"]
        if not load_teacher:
            sd = {k: v for k, v in sd.items() if not k.startswith("teacher")}
        want = set(model.state_dict())
        sd = {(k if k in want else k.replace(".vision_model.", ".", 1)): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        return model

    TextAlignedTokenizer.from_checkpoint = from_checkpoint
    TextAlignedTokenizer._tar_rlvr_patched = True


class TATok:
    def __init__(self, device: str = "cpu", tar_root: str = "/mnt/work/Tar"):
        if tar_root not in sys.path:
            sys.path.insert(0, tar_root)
        import torch
        from huggingface_hub import hf_hub_download
        from tok.mm_autoencoder import MMAutoEncoder
        from tar_rlvr.vis.safe_load import restricted_torch_load
        self.torch = torch
        _patch_ta_tok_keys()
        with restricted_torch_load():
            self.m = MMAutoEncoder(
                ar_path=hf_hub_download("ByteDance-Seed/Tar-TA-Tok", "ar_dtok_lp_256px.pth"),
                encoder_path=hf_hub_download("ByteDance-Seed/Tar-TA-Tok", "ta_tok.pth"),
                decoder_path=hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt"),
                encoder_args={"input_type": "rec"}, decoder_args={},
            ).eval().to(dtype=torch.float32, device=device)
        self.device = device
        torch.set_grad_enabled(False)

    def decode(self, scale: int, codes, cfg: float = 4.0, seed: int = 0):
        """시각 토큰 → PIL 이미지 (256px)."""
        from PIL import Image
        n = SEQ[scale]
        codes = list(codes)[:n] + [0] * max(0, n - len(codes))
        self.torch.manual_seed(seed)
        self.m.ar_model.cls_token_num = n
        self.m.encoder.pool_scale = scale + 1
        img = self.m.decode_from_encoder_indices(self.torch.tensor(codes).unsqueeze(0).to(self.device), {"cfg_scale": cfg})
        return Image.fromarray(np.asarray(img[0]))
