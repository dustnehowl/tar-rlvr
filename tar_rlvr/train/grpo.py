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
    if m.get("local_path"):                       # 이미 vision_tower 를 뺀 safetensors 디렉토리
        path, stripped = m["local_path"], True
    else:
        from huggingface_hub import snapshot_download
        path, stripped = snapshot_download(m["hf_id"], ignore_patterns=["*.pth"]), False
    model, _ = tar_lib.load_tar7b(path, mesh, remat=m.get("remat", False), already_stripped=stripped)
    if lora.get("enabled"):
        model = apply_lora(model, mesh, lora)
    tok_path = m.get("tokenizer_path") or path
    return model, tar_lib.load_tokenizer(tok_path)


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
    # 새로 생긴 LoRA 파라미터만 샤딩한다. 전체 state 에 걸면 backbone 배열이 새로 만들어져
    # reference(=base) 와 backbone 공유가 깨진다 → KL 기록용 reference 가 7B 를 통째로 한 벌 더 쓰게 됨.
    with mesh:
        state = nnx.state(lora_model, nnx.LoRAParam)
        nnx.update(lora_model, jax.lax.with_sharding_constraint(state, nnx.get_partition_spec(state)))
    return lora_model


def format_prompt(tok, text: str) -> str:
    """Instruct 모델은 chat template 없이는 바로 EOS 를 내버린다. HF 토크나이저면 템플릿 적용."""
    if not hasattr(tok, "apply_chat_template"):
        return text   # tiny ByteTokenizer
    msgs = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_dataset(cfg, batch_size: int, tok, max_steps: int | None = None, return_eval: bool = False):
    """return_eval=True 면 (batches, eval_rows). eval.n_prompts 개를 고정 시드로 떼어 학습에서 뺀다 (held-out)."""
    d = cfg["data"]
    rows = make_synthetic(d["n_samples"], instruction_kind=d.get("instruction", "sketch")) if d["source"] == "synthetic" else None
    if rows is None:
        from tar_rlvr.data.cova import load_cova
        rows = load_cova(d)
    for r in rows:
        r["prompts"] = format_prompt(tok, r["prompts"])
    ev = cfg.get("eval") or {}
    n_eval = min(int(ev.get("n_prompts") or 0), max(0, len(rows) - batch_size))
    eval_rows = []
    if n_eval:
        import random as _random
        order = sorted(rows, key=lambda r: str(r.get("qid", r["prompts"])))
        _random.Random(ev.get("seed", 4242)).shuffle(order)
        eval_rows, held = order[:n_eval], {id(r) for r in order[:n_eval]}
        rows = [r for r in rows if id(r) not in held]
        print(f"[data] held-out eval: {len(eval_rows)} rows (학습에서 제외) → train {len(rows)} rows")
    # 값은 np.array 여야 한다: Tunix 가 jax.tree.map(np.repeat) 로 G 배 복제하는데 list[str] 이면 구조가 깨진다.
    # Tunix 는 train_ds 를 한 번만 순회하고 멈춘다 → max_steps 만큼 되도록 에폭 반복 (에폭마다 재셔플)
    import math, random, numpy as np
    n_batches = len(rows) // batch_size
    epochs = max(1, math.ceil(max_steps / n_batches)) if max_steps else 1
    batches = []
    for ep in range(epochs):
        order = list(rows)
        random.Random(1000 + ep).shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            chunk = order[i:i + batch_size]
            batches.append({k: np.array([r[k] for r in chunk]) for k in chunk[0]})
    print(f"[data] {len(rows)} rows × {epochs} epochs = {len(batches)} batches of {batch_size} (max_steps={max_steps})")
    return (batches, eval_rows) if return_eval else batches


def train(cfg: dict, max_steps: int | None = None, debug: bool = False, run_name: str | None = None):
    """flax eager-sharding 은 변수 생성 시 mesh 컨텍스트를 요구 → 전체를 set_mesh 안에서."""
    from tunix.utils import compat
    mesh = build_mesh(cfg["mesh"])
    with compat.set_mesh(mesh):
        return _train(cfg, mesh, max_steps, debug, run_name)


def _train(cfg, mesh, max_steps, debug, run_name=None):
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo import grpo_learner as grpo_lib
    from tunix.rl.rollout import base_rollout
    from tar_rlvr.train.continuation import install_continuation, install_bridge_mask, visual_metrics

    g = cfg["grpo"]
    steps = max_steps or g["max_steps"]
    tcfg = cfg.get("telemetry", {})
    want_ref = g["beta"] > 0 or tcfg.get("log_kl", False)
    if want_ref and cfg["model"].get("lora", {}).get("enabled"):
        # LoRA 면 reference = LoRA 를 씌우기 전 base. 같은 backbone 을 공유해 가중치 복사가 생기지 않는다.
        reference, tok = build_model_and_tokenizer(cfg, mesh, with_lora=False)
        policy = apply_lora(reference, mesh, cfg["model"]["lora"])
    else:
        policy, tok = build_model_and_tokenizer(cfg, mesh)
        reference = build_model_and_tokenizer(cfg, mesh, with_lora=False)[0] if want_ref else None
    dataset, eval_rows = build_dataset(cfg, g["batch_size"], tok, max_steps=steps, return_eval=True)
    reward_fns = build_reward_fns(cfg["rewards"])
    metric_fns = [visual_metrics]          # 시각 사용률 등 핵심 지표는 항상 기록
    import time as _time
    run_name = run_name or _time.strftime("run_%Y%m%d_%H%M%S")
    tel = None
    if tcfg.get("enabled", True):
        from tar_rlvr.train.telemetry import Telemetry
        tel = Telemetry(cfg, tok, run_dir=f"logs/runs/{run_name}")
        metric_fns.append(tel.metric_fn)
        print(f"[telemetry] logs/runs/{run_name}/ 에 기록 — 업로드: scripts/wandb_sync.py logs/runs/{run_name}")
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
            # checkpoint_dir/<run_name>. 같은 run 이름으로 다시 실행하면 Tunix 가 마지막 체크포인트에서 이어간다
            # (데이터도 그만큼 건너뜀). VM 이 꺼져도 run 을 살리기 위한 것.
            checkpoint_root_directory=f"{cfg['checkpoint_dir'].rstrip('/')}/{run_name}" if cfg.get("checkpoint_dir") else None,
            checkpointing_options=_ckpt_options(cfg.get("checkpoint_every_secs", 900)) if cfg.get("checkpoint_dir") else None,
            metrics_logging_options=tel.metrics_logging_options() if tel else None,
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
        # beta=0 이면 KL 은 기록만 한다 → 음수가 안 나오는 k3 추정치(low_var_kl). beta>0 이면 기존 기본값 유지.
        algo_config=grpo_lib.GRPOConfig(num_generations=g["num_generations"], beta=g["beta"], epsilon=g["epsilon"],
                                        kl_loss_mode=g.get("kl_loss_mode", "low_var_kl" if g["beta"] == 0 else "kl"),
                                        loss_agg_mode=g.get("loss_agg_mode", "sequence-mean-token-mean")),
        reward_fns=reward_fns,
        metric_fns=metric_fns,
    )
    if cont.get("enabled", True) and cont.get("mask_bridge", False):
        install_bridge_mask(learner)
    elif cont.get("mask_bridge", False):
        raise ValueError("continuation.mask_bridge 는 continuation.enabled 와 같이 쓴다")
    fi = g.get("forced_image") or {}
    if fi.get("enabled"):
        if cont.get("enabled", True):
            raise ValueError("forced_image 는 continuation 과 같이 쓰지 않는다 (그림 뒤 EOS 에 텍스트를 붙이지 않는 실험)")
        from tar_rlvr.train import forced_image
        forced_image.install(learner, g["num_generations"], fi.get("per_group", 2), fi.get("prefix", forced_image.DEFAULT_PREFIX))
    if tel:
        tel.attach(cluster, learner, eval_rows)
    # Tunix 는 actor 를 내부 복사본으로 학습하므로 반드시 actor_trainer.model 을 봐야 한다
    before = _lora_snapshot(cluster.actor_trainer.model) if debug else None
    learner.train(dataset)
    if debug:
        after = _lora_snapshot(cluster.actor_trainer.model)
        delta = float(jnp.sqrt(sum(jnp.sum(jnp.square(a - b)) for a, b in zip(after, before))))
        print(f"\n[check] LoRA update L2 = {delta:.3e}  (lr={g['learning_rate']}; 0 이면 gradient 가 안 흐르는 것)")
    return learner


def _ckpt_options(every_secs: int):
    import orbax.checkpoint as ocp
    return ocp.CheckpointManagerOptions(
        save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(minimum_interval_secs=every_secs),
        max_to_keep=2)


def _lora_snapshot(model):
    return [x.astype(jnp.float32) for x in jax.tree.leaves(nnx.state(model, nnx.LoRAParam))]