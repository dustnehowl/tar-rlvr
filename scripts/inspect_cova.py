"""math_vr / mathnet 스키마·답 분포 확인. 답 길이 분포를 보고 configs 의 max_answer_chars 를 정한다."""
import json, collections
from datasets import load_dataset
for name in ["math_vr", "mathnet"]:
    ds = load_dataset("tsunghanwu/cova", name, split="train")
    ds = ds.remove_columns(["images", "interleaved_reasoning"])
    print(f"\n===== {name}: {len(ds)} rows, columns={ds.column_names}")
    lens = collections.Counter()
    for r in ds:
        a = str(r["answer"]); lens[min(len(a) // 10 * 10, 200)] += 1
    print("answer 길이 분포 (10자 단위):", sorted(lens.items()))
    for r in ds.select(range(3)):
        print("--- question:", r["question"][:300].replace("\n", " "))
        print("--- answer:", r["answer"][:200])
        print("--- metadata:", r["metadata"][:300])
