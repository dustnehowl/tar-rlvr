"""GRPO 학습 진입점. Tunix 의 RLCluster + GrpoLearner 를 조립한다.

NOTE: Tunix RL API 는 버전 간 이름이 바뀐다 (GRPOConfig/GrpoConfig, RLTrainingConfig 등).
설치 후 반드시 확인:
  python -c "import tunix.rl.grpo.grpo_learner as g; print([n for n in dir(g) if 'onfig' in n or 'Learner' in n])"
  python -c "import tunix.rl.rl_cluster as r; print([n for n in dir(r) if 'Config' in n or 'Cluster' in n])"
이 파일의 역할은 '우리 쪽 로직(데이터/리워드/모델)'을 한 곳에서 Tunix에 꽂는 것이라, 어댑터가 깨지면 여기만 고치면 된다.
"""
from __future__ import annotations
import jax, optax
from flax import nnx

from tar_rlvr.rewards import build_reward_fns
from tar_rlvr.data.synthetic import make_synthetic
from tar_rlvr.model import tar as tar_lib


def build_mesh(spec):
    shape, axes = tuple(spec[0]), tuple(spec[1])
    return jax.make_mesh(shape, axes, axis_types=(jax.sharding.AxisType.Auto,) * len(shape))


def build_model_and_tokenizer(cfg, mesh):
    m = cfg["model"]
    if m["kind"] == "tiny":
        from tar_rlvr.model.tiny_tokenizer import ByteTokenizer
        model, mcfg = tar_lib.make_tiny(mesh, m["num_visual_tokens"])
        return model, mcfg, ByteTokenizer(m["num_visual_tokens"])
    from huggingface_hub import snapshot_download
    path = snapshot_download(m["hf_id"], ignore_patterns=["*.pth"])
    model, mcfg = tar_lib.load_tar7b(path, mesh)
    if m.get("lora", {}).get("enabled"):
        model = apply_lora(model, mesh, m["lora"])
    return model, mcfg, tar_lib.load_tokenizer(path)


def apply_lora(model, mesh, lora_cfg):
    import qwix
    provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=lora_cfg["rank"], alpha=lora_cfg["alpha"],
    )
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(model, provider, **model_input)
    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        nnx.update(lora_model, jax.lax.with_sharding_constraint(state, pspecs))
    return lora_model


def build_dataset(cfg):
    d = cfg["data"]
    if d["source"] == "synthetic":
        return make_synthetic(d["n_samples"], d["tasks"])
    from tar_rlvr.data.cova import load_cova
    return load_cova(d)


def train(cfg: dict, max_steps: int | None = None):
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo import grpo_learner as grpo_lib
    from tunix.generate import sampler as sampler_lib  # noqa: F401  (vLLM 롤아웃 전환 시 교체)

    g = cfg["grpo"]
    steps = max_steps or g["max_steps"]
    mesh = build_mesh(cfg["mesh"])
    policy, mcfg, tok = build_model_and_tokenizer(cfg, mesh)
    ref_model, _, _ = build_model_and_tokenizer(cfg, mesh)  # KL용 reference (beta>0 일 때만 필요)
    dataset = build_dataset(cfg)

    cluster_cfg = rl_cluster_lib.ClusterConfig(
        role_to_mesh={r: mesh for r in rl_cluster_lib.Role},
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.adamw(g["learning_rate"]),
            eval_every_n_steps=g.get("eval_every", 10),
            max_steps=steps,
            mini_batch_size=g["batch_size"],
        ),
        rollout_config=rl_cluster_lib.RolloutConfig(
            max_tokens_to_generate=g["max_completion_length"],
            max_prompt_length=g["max_prompt_length"],
            temperature=1.0,
        ),
    )
    cluster = rl_cluster_lib.RLCluster(actor=policy, reference=ref_model, tokenizer=tok, cluster_config=cluster_cfg)
    learner = grpo_lib.GRPOLearner(
        rl_cluster=cluster,
        reward_fns=build_reward_fns(cfg["rewards"]),
        grpo_config=grpo_lib.GRPOConfig(
            num_generations=g["num_generations"], beta=g["beta"], epsilon=g["epsilon"],
        ),
    )
    learner.train(dataset)
    return learner
