"""Tar-7B 를 Tunix(Flax NNX) Qwen2 구현으로 로드.

Tar-7B = Qwen2.5-7B-Instruct + 65,536 시각 토큰 → vocab 만 확장된 safetensors.
따라서 Tunix qwen2 ModelConfig 의 vocab_size 만 체크포인트에 맞춰 덮어쓰면 된다.

NOTE: Tunix API 는 빠르게 바뀐다. 설치된 버전에서 아래 import 경로/필드명을 확인할 것:
  python -c "from tunix.models.qwen2 import model, params_safetensors; help(model.ModelConfig)"
"""
from __future__ import annotations
import json, os
import jax
import jax.numpy as jnp


def tar_vocab_size(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)["vocab_size"]


def load_tar7b(model_path: str, mesh, dtype=jnp.bfloat16):
    from tunix.models.qwen2 import model as qwen2_lib
    from tunix.models.qwen2 import params_safetensors as params_lib
    import dataclasses
    cfg = qwen2_lib.ModelConfig.qwen2_5_7b()
    cfg = dataclasses.replace(cfg, vocab_size=tar_vocab_size(model_path))
    with mesh:
        model = params_lib.create_model_from_safe_tensors(model_path, cfg, mesh, dtype=dtype)
    return model, cfg


def load_tokenizer(model_path: str):
    """HF 토크나이저. Tar 체크포인트에는 <I0>..<I65535> 가 added_tokens 로 들어있어야 한다 — 검증."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    assert tok.convert_tokens_to_ids("<I0>") != tok.unk_token_id, "visual tokens missing in tokenizer"
    return tok


def make_tiny(mesh, num_visual_tokens: int = 64, text_vocab: int = 256):
    """CPU 스모크용 랜덤 초기화 소형 Qwen2 (+시각 토큰)."""
    from tunix.models.qwen2 import model as qwen2_lib
    from flax import nnx
    import dataclasses
    base = qwen2_lib.ModelConfig.qwen2_5_0_5b()
    cfg = dataclasses.replace(
        base, vocab_size=text_vocab + num_visual_tokens, num_layers=2,
        embed_dim=64, hidden_dim=128, num_heads=2, num_kv_heads=1, head_dim=32,
    )
    with mesh:
        model = qwen2_lib.Qwen2(cfg, rngs=nnx.Rngs(0))
    return model, cfg
