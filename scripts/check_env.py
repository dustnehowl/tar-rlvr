"""TPU/Tunix 환경 점검: 디바이스 수, Tunix 버전, 우리가 의존하는 심볼이 실제로 있는지."""
import importlib, jax
print("devices:", jax.device_count(), jax.devices()[0].platform)
import tunix; print("tunix:", getattr(tunix, "__version__", "?"))
checks = {
    "tunix.models.qwen2.model": ["ModelConfig", "Qwen2"],
    "tunix.models.qwen2.params_safetensors": ["create_model_from_safe_tensors"],
    "tunix.rl.rl_cluster": ["RLCluster", "ClusterConfig", "RLTrainingConfig", "RolloutConfig", "Role"],
    "tunix.rl.grpo.grpo_learner": ["GRPOLearner", "GRPOConfig"],
}
ok = True
for mod, names in checks.items():
    try:
        m = importlib.import_module(mod)
        for n in names:
            if not hasattr(m, n):
                ok = False; print(f"MISSING {mod}.{n}  -> candidates: {[x for x in dir(m) if n[:4].lower() in x.lower()]}")
    except Exception as e:
        ok = False; print(f"IMPORT FAIL {mod}: {e}")
print("ALL OK" if ok else "FIX tar_rlvr/train/grpo.py adapter before training")
