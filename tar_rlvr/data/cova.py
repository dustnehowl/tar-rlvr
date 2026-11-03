"""CoVA (tsunghanwu/cova) 로더. 스키마는 scripts/inspect_cova.py 로 먼저 확인하고
configs/*.yaml 의 prompt_column/answer_column/task_column 을 채울 것.

TODO(스키마 확인 후):
  - task 이름 → VERIFIERS 키 매핑 (TASK_MAP)
  - task별 ref 구성 (sudoku는 puzzle, coloring은 edges/k 가 필요)
"""
from __future__ import annotations
import json
from .synthetic import SYSTEM

# CoVA task 이름 → verifier 키. 실제 이름 확인 후 수정.
TASK_MAP = {
    "arc": "arc", "maze": "maze", "sudoku": "sudoku",
    "graph_coloring": "graph_coloring", "clutrr": "exact", "prosqa": "exact",
}


def load_cova(cfg: dict) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(cfg["hf_id"], split=cfg["split"])
    if cfg.get("tasks"):
        ds = ds.filter(lambda r: r[cfg["task_column"]] in cfg["tasks"])
    rows = []
    for r in ds:
        task = TASK_MAP.get(r[cfg["task_column"]])
        if task is None:
            continue
        rows.append({
            "prompts": f"{SYSTEM}\n{r[cfg['prompt_column']]}",
            "task": task,
            "ref": json.dumps({"answer": r[cfg["answer_column"]]}),  # TODO: task별 ref 확장
        })
    return rows
