"""CoVA 스키마 확인: 컬럼/태스크 분포/샘플 1개 출력."""
from collections import Counter
from datasets import load_dataset
ds = load_dataset("tsunghanwu/cova", split="train", streaming=True)
first = next(iter(ds))
print("columns:", list(first.keys()))
for k, v in first.items():
    print(f"--- {k} ---\n{str(v)[:600]}")
