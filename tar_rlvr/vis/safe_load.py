"""TA-Tok / LlamaGen 체크포인트를 허용 목록 unpickler 로 읽는다.

체크포인트 pickle 은 easydict.EasyDict 를 담고 있어 torch>=2.6 의 weights_only=True 로는 안 읽힌다
(EasyDict 의 SETITEMS 를 weights_only unpickler 가 거부). weights_only 를 끄는 대신, pickletools 로 정적
확인한 4개 전역(텐서 복원, OrderedDict, EasyDict)만 허용하는 Unpickler 를 쓴다. 그 밖의 전역이 나오면 실패한다.
"""
from __future__ import annotations
import contextlib, pickle, types

ALLOWED = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "FloatStorage"),
    ("easydict", "EasyDict"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"허용되지 않은 전역: {module}.{name}")
        return super().find_class(module, name)


restricted_pickle = types.ModuleType("restricted_pickle")
restricted_pickle.Unpickler = _RestrictedUnpickler
restricted_pickle.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()
restricted_pickle.__name__ = "restricted_pickle"


@contextlib.contextmanager
def restricted_torch_load():
    """이 블록 안의 torch.load 는 허용 목록 unpickler 를 쓴다 (Tar 코드가 내부에서 torch.load 를 부르므로)."""
    import torch
    orig = torch.load

    def load(f, *a, **kw):
        kw.pop("weights_only", None)
        kw["pickle_module"] = restricted_pickle
        return orig(f, *a, weights_only=False, **kw)

    torch.load = load
    try:
        yield
    finally:
        torch.load = orig
