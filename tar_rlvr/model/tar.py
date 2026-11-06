"""Tar-7B 를 Tunix(Flax NNX) Qwen2 구현으로 로드. (Tunix 0.1.5 API 기준)

Tar-7B = Qwen2.5-7B-Instruct + 65,536 시각 토큰 + SigLIP 비전 타워.
Tunix safetensors 로더는 모르는 키(model.vision_tower.*)를 만나면 죽으므로,
LLM 키만 추린 복사본(<model_path>/llm_only/)을 한 번 만들어 두고 그걸 로드한다.
비전 타워는 RL 에 필요 없다 (CoVA 입력은 텍스트).
"""
from __future__ import annotations
import dataclasses, glob, json, os, re, shutil
import jax.numpy as jnp

# Tunix qwen2 key mapping 이 아는 키들 (tunix/models/qwen2/params.py 와 동일해야 함)
_LLM_KEY = re.compile(
    r"^(model\.embed_tokens\.weight|model\.norm\.weight|lm_head\.weight|"
    r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj\.(weight|bias)|mlp\.(gate|up|down)_proj\.weight|"
    r"(input|post_attention)_layernorm\.weight))$"
)


def tar_vocab_size(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)["vocab_size"]


def prepare_llm_only(model_path: str) -> str:
    """vision tower 등 비-LLM 키를 뺀 safetensors 복사본을 만든다 (이미 있으면 재사용)."""
    out = os.path.join(model_path, "llm_only")
    if os.path.exists(os.path.join(out, "done")):
        return out
    from safetensors import safe_open
    from safetensors.numpy import save_file
    os.makedirs(out, exist_ok=True)
    kept, dropped = 0, []
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        tensors = {}
        with safe_open(f, framework="np") as sf:
            for k in sf.keys():
                if _LLM_KEY.match(k):
                    tensors[k] = sf.get_tensor(k); kept += 1
                else:
                    dropped.append(k)
        if tensors:
            save_file(tensors, os.path.join(out, os.path.basename(f)), metadata={"format": "pt"})
        print(f"[tar] {os.path.basename(f)}: kept {len(tensors)} tensors")
    for name in ("config.json", "generation_config.json"):
        if os.path.exists(os.path.join(model_path, name)):
            shutil.copy(os.path.join(model_path, name), out)
    with open(os.path.join(out, "dropped_keys.txt"), "w") as fh:
        fh.write("\n".join(dropped))
    open(os.path.join(out, "done"), "w").close()
    print(f"[tar] llm_only ready: kept {kept}, dropped {len(dropped)} (see dropped_keys.txt)")
    return out


def load_tar7b(model_path: str, mesh, dtype=jnp.bfloat16, remat: bool = True):
    from tunix.models.qwen2 import model as qwen2_lib
    from tunix.models.qwen2 import params as params_lib
    llm_path = prepare_llm_only(model_path)
    cfg = dataclasses.replace(
        qwen2_lib.ModelConfig.qwen2_5_7b(),
        vocab_size=tar_vocab_size(model_path),
        # BLOCK: attention 블록 활성화를 backward 때 재계산 → 메모리 ↓, 시간 ~+30%
        remat_config=qwen2_lib.RematConfig.BLOCK if remat else qwen2_lib.RematConfig.NONE,
    )
    with mesh:
        model = params_lib.create_model_from_safe_tensors(llm_path, cfg, mesh, dtype=dtype)
    return model, cfg


def load_tokenizer(model_path: str):
    """HF 토크나이저. Tar 체크포인트에 <I0>..<I65535> 가 added_tokens 로 있어야 한다 — 검증."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    assert tok.convert_tokens_to_ids("<I0>") not in (None, tok.unk_token_id), "visual tokens missing in tokenizer"
    return tok


def make_tiny(mesh, vocab_size: int):
    """스모크용 랜덤 초기화 소형 Qwen2."""
    from tunix.models.qwen2 import model as qwen2_lib
    from tunix.utils import compat
    from flax import nnx
    cfg = qwen2_lib.ModelConfig(
        num_layers=2, vocab_size=vocab_size, embed_dim=64, hidden_dim=128,
        num_heads=2, head_dim=32, num_kv_heads=1, rope_theta=10_000, norm_eps=1e-6,
    )
    with compat.set_mesh(mesh):   # flax eager-sharding 요구사항 (Tunix safetensors_loader 와 동일)
        model = qwen2_lib.Qwen2(cfg, rngs=nnx.Rngs(0))
    return model, cfg