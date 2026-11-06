"""환경 점검: 디바이스, Tunix 버전, grpo.py 가 의존하는 심볼 존재 여부 (Tunix 0.1.5 기준)."""
import importlib, jax
print("devices:", jax.device_count(), jax.devices()[0].platform)
import tunix; print("tunix:", getattr(tunix, "__version__", "?"))
checks = {
    "tunix.models.qwen2.model": ["ModelConfig", "Qwen2"],
    "tunix.models.qwen2.params": ["create_model_from_safe_tensors"],
    "tunix.rl.rl_cluster": ["RLCluster", "ClusterConfig", "RLTrainingConfig", "Role"],
    "tunix.rl.rollout.base_rollout": ["RolloutConfig"],
    "tunix.rl.grpo.grpo_learner": ["GRPOLearner", "GRPOConfig"],
}
ok = True
for mod, names in checks.items():
    try:
        m = importlib.import_module(mod)
        for n in names:
            if not hasattr(m, n):
                ok = False; print(f"MISSING {mod}.{n} -> candidates: {[x for x in dir(m) if n[:4].lower() in x.lower()]}")
    except Exception as e:
        ok = False; print(f"IMPORT FAIL {mod}: {e}")
print("ALL OK" if ok else "FIX tar_rlvr/train/grpo.py adapter before training")
