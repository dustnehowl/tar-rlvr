"""GRPO 학습 진입점 — Tunix 0.1.5 RLCluster + GRPOLearner 어댑터.
API 바뀌면 이 파일만 고친다. scripts/check_env.py 가 심볼 존재를 먼저 확인한다.

Tunix 데이터 규약: train_ds 는 *배치* dict 의 iterable. 각 배치 = {"prompts": [str,...], 그외 컬럼: [...]}.
추가 컬럼(task, ref)은 reward_fn 에 **kwargs 로 그대로 전달된다.
"""
from __future__ import annotations
import jax, optax
import jax.numpy as jnp
from flax import nnx

from tar_rlvr.rewards import build_reward_fns
from tar_rlvr.data.synthetic import make_synthetic
from tar_rlvr.model import tar as tar_lib

CHAT_SYSTEM = "You are a helpful assistant."


def _pow2(n):
    return 1 << (n - 1).bit_length()


def build_mesh(spec):
    shape, axes = tuple(spec[0]), tuple(spec[1])
    return jax.make_mesh(shape, axes, axis_types=(jax.sharding.AxisType.Auto,) * len(shape))


def build_model_and_tokenizer(cfg, mesh, with_lora: bool = True):
    m = cfg["model"]
    lora = m.get("lora", {}) if with_lora else {}
    if m["kind"] == "tiny":
        from tar_rlvr.model.tiny_tokenizer import ByteTokenizer
        tok = ByteTokenizer(m["num_visual_tokens"])
        model, _ = tar_lib.make_tiny(mesh, tok.vocab_size)
        if lora.get("enabled"):
            model = apply_lora(model, mesh, lora)
        return model, tok
    from huggingface_hub import snapshot_download
    path = snapshot_download(m["hf_id"], ignore_patterns=["*.pth"])
    model, _ = tar_lib.load_tar7b(path, mesh, remat=m.get("remat", False))
    if lora.get("enabled"):
        model = apply_lora(model, mesh, lora)
    return model, tar_lib.load_tokenizer(path)


def apply_lora(model, mesh, lora_cfg):
    import qwix
    provider = qwix.LoraProvider(
        module_path=".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
        rank=lora_cfg["rank"], alpha=lora_cfg["alpha"],
    )
    # rngs 필수: 없으면 LoRA A/B 가 둘 다 0 으로 초기화돼 gradient 가 0 → 영원히 안 배움.
    lora_model = qwix.apply_lora_to_model(
        model, provider, **model.get_model_input(), rngs=nnx.Rngs(lora_cfg.get("seed", 0))
    )
    # qwix 동적 서브클래스가 flax 0.12 에서 pytree=True 로 생성돼 nnx.state() 가 비는 문제 우회
    type(lora_model)._pytree__is_pytree = False
    with mesh:
        state = nnx.state(lora_model)
        nnx.update(lora_model, jax.lax.with_sharding_constraint(state, nnx.get_partition_spec(state)))
    return lora_model


def format_prompt(tok, text: str) -> str:
    """Instruct 모델은 chat template 없이는 바로 EOS 를 내버린다. HF 토크나이저면 템플릿 적용."""
    if not hasattr(tok, "apply_chat_template"):
        return text   # tiny ByteTokenizer
    msgs = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_dataset(cfg, batch_size: int, tok):
    d = cfg["data"]
    rows = make_synthetic(d["n_samples"]) if d["source"] == "synthetic" else None
    if rows is None:
        from tar_rlvr.data.cova import load_cova
        rows = load_cova(d)
    for r in rows:
        r["prompts"] = format_prompt(tok, r["prompts"])
    # 값은 np.array 여야 한다: Tunix 가 jax.tree.map(np.repeat) 로 G 배 복제하는데 list[str] 이면 구조가 깨진다.
    import numpy as np
    batches = []
    for i in range(0, len(rows) - batch_size + 1, batch_size):
        chunk = rows[i:i + batch_size]
        batches.append({k: np.array([r[k] for r in chunk]) for k in chunk[0]})
    return batches


def train(cfg: dict, max_steps: int | None = None, debug: bool = False):
    """flax eager-sharding 은 변수 생성 시 mesh 컨텍스트를 요구 → 전체를 set_mesh 안에서."""
    from tunix.utils import compat
    mesh = build_mesh(cfg["mesh"])
    with compat.set_mesh(mesh):
        return _train(cfg, mesh, max_steps, debug)


def _train(cfg, mesh, max_steps, debug):
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo import grpo_learner as grpo_lib
    from tunix.rl.rollout import base_rollout
    from tar_rlvr.train.continuation import install_continuation, visual_metrics

    g = cfg["grpo"]
    steps = max_steps or g["max_steps"]
    policy, tok = build_model_and_tokenizer(cfg, mesh)
    reference, _ = build_model_and_tokenizer(cfg, mesh, with_lora=False) if g["beta"] > 0 else (None, None)
    dataset = build_dataset(cfg, g["batch_size"], tok)
    reward_fns = build_reward_fns(cfg["rewards"])
    metric_fns = [visual_metrics]          # 시각 사용률 등 핵심 지표는 항상 기록
    if debug:
        from tar_rlvr.train import debug_log
        debug_log.banner("[0. 토크나이저]")
        for name in ("bos_token", "eos_token", "pad_token"):
            print(f"{name} = {getattr(tok, name, None)!r}")
        debug_log.banner("[1. 데이터] 첫 배치")
        for k, v in dataset[0].items():
            print(f"--- {k}\n{v[0]}")
        reward_fns[0] = debug_log.make_logging_reward(reward_fns[0])
        metric_fns.append(debug_log.make_logging_metric([f.__name__ for f in reward_fns], g["num_generations"]))

    cluster_cfg = rl_cluster_lib.ClusterConfig(
        role_to_mesh={r: mesh for r in rl_cluster_lib.Role},
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.adamw(g["learning_rate"]),
            eval_every_n_steps=g.get("eval_every", 10),
            max_steps=steps,
            mini_batch_size=g["batch_size"],   # 프롬프트 수 기준 (generations 곱하지 않음)
            train_micro_batch_size=g.get("train_micro_batch_size"),
            compute_logps_micro_batch_size=g.get("logps_micro_batch_size"),
            rollout_micro_batch_size=g.get("rollout_micro_batch_size"),
            checkpoint_root_directory=cfg.get("checkpoint_dir"),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=g["max_completion_length"],
            max_prompt_length=g["max_prompt_length"],
            kv_cache_size=_pow2(g["max_prompt_length"] + g["max_completion_length"]),
            temperature=g.get("temperature", 1.0),
        ),
    )
    cluster = rl_cluster_lib.RLCluster(actor=policy, reference=reference, tokenizer=tok, cluster_config=cluster_cfg)
    cont = g.get("continuation", {})
    if cont.get("enabled", True):
        install_continuation(cluster, tok, max_completion=g["max_completion_length"],
                             min_cont=cont.get("min_tokens", 96),
                             bridge=cont.get("bridge", "\nBased on the sketch, "), debug=debug)
    learner = grpo_lib.GRPOLearner(
        rl_cluster=cluster,
        algo_config=grpo_lib.GRPOConfig(num_generations=g["num_generations"], beta=g["beta"], epsilon=g["epsilon"]),
        reward_fns=reward_fns,
        metric_fns=metric_fns,
    )
    # Tunix 는 actor 를 내부 복사본으로 학습하므로 반드시 actor_trainer.model 을 봐야 한다
    before = _lora_snapshot(cluster.actor_trainer.model) if debug else None
    learner.train(dataset)
    if debug:
        after = _lora_snapshot(cluster.actor_trainer.model)
        delta = float(jnp.sqrt(sum(jnp.sum(jnp.square(a - b)) for a, b in zip(after, before))))
        print(f"\n[check] LoRA update L2 = {delta:.3e}  (lr={g['learning_rate']}; 0 이면 gradient 가 안 흐르는 것)")
    return learner


def _lora_snapshot(model):
    return [x.astype(jnp.float32) for x in jax.tree.leaves(nnx.state(model, nnx.LoRAParam))]