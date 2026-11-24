"""학습 로그(--debug)의 롤아웃을 현재 verifier 로 재채점해서 거짓 음성(false negative)을 센다.

사용: python scripts/rescore_log.py logs/tpu_run3.log [--config configs/tpu_v6e_7b.yaml] [--out logs/tpu_run3.rescored.jsonl]

로그에는 ref 가 첫 배치에만 찍히므로, 데이터셋을 config 대로 다시 불러와 프롬프트 텍스트로 매칭한다
(기본은 HF 오프라인 — 캐시 사용). 각 completion 을 세 기준으로 채점한다:
  train : 학습 당시 재현 — 옛 verifier + 옛 BOXED_RE + math-verify 기본 timeout + 워커 스레드 (기록과 일치해야 함)
  mv    : 옛 verifier + 옛 BOXED_RE, math-verify 정상 → train 과의 차이 = 스레드 버그 몫
  boxed : 옛 verifier + 새 boxed 파서, math-verify 정상 → mv 와의 차이 = boxed 중첩 버그 몫
  new   : 현재 코드                                     → boxed 와의 차이 = verifier 비교 규칙 변경 몫
0→1 로 바뀐 것 중 근사 비교(_numeric_close)에만 기대거나, 객관식인데 글자가 아닌 답으로 맞은 것은
suspect 로 표시한다 (확인 권장).
로그는 1200자를 넘는 completion 의 가운데를 생략하므로 그 샘플은 truncated 로 표시한다.
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLD_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")   # 수정 전 정규식 (1단계 중첩까지)
LETTER = re.compile(r"^\(?([A-H])\)?[.)]?$")


def parse_log(path):
    """[2. 롤아웃] 블록마다 (prompt, completions[8], logged_rewards[8])."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = []
    for b in re.split(r"\[2\. 롤아웃\] step \d+", txt)[1:]:
        roll, _, rew = b.partition("[3→4. reward 합산")
        prompt = roll.split("--- prompt (공통)", 1)[1].split("\n#0\n")[0]
        comps = [c.split("\n=====")[0].rstrip("\n") for c in re.split(r"\n#\d+\n--- completion\n", roll)[1:]]
        logged = [float(x) for x in re.findall(r"샘플 \d+: reward=([+-]\d+\.\d+)", rew)]
        if len(comps) != len(logged):
            raise ValueError(f"블록 {len(blocks) + 1}: completion {len(comps)}개 vs reward {len(logged)}개")
        blocks.append((prompt, comps, logged))
    return blocks


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def match_refs(blocks, rows):
    """로그 프롬프트(앞뒤 생략·시각 토큰 접힘)에 row 프롬프트의 앞/뒤 150자가 들어있는지로 매칭."""
    from tar_rlvr.train.debug_log import _s
    keys = [(_norm(_s(r["prompts"]))[:150], _norm(_s(r["prompts"]))[-150:], r) for r in rows]
    out = []
    for i, (p, _, _) in enumerate(blocks):
        p = _norm(p)
        hits = {r["qid"]: r for h, t, r in keys if h in p and t in p}
        if len(hits) != 1:
            raise ValueError(f"블록 {i + 1}: 데이터셋 매칭 {len(hits)}건 (1건이어야 함)")
        out.append(next(iter(hits.values())))
    return out


def _score_all(items, *, old_boxed: bool, mv_default_timeout: bool, legacy: bool = False,
               numeric_close: bool = True, in_thread: bool):
    """items: [(completion, ref_json)] → [(reward, extracted)]. 전역 설정을 잠깐 바꿔 각 기준을 재현한다."""
    from tar_rlvr.rewards import parsing, verifiers
    from tar_rlvr.rewards import correctness_reward
    saved = (parsing.boxed_contents, verifiers._MV_KW_PARSE, verifiers._MV_KW_VERIFY, verifiers._numeric_close,
             verifiers.VERIFIERS["math"])
    if legacy:
        verifiers.VERIFIERS["math"] = verifiers.legacy_math_match
    if old_boxed:
        parsing.boxed_contents = lambda t: OLD_BOXED_RE.findall(t)
    if mv_default_timeout:
        verifiers._MV_KW_PARSE, verifiers._MV_KW_VERIFY = {}, {}
    if not numeric_close:
        verifiers._numeric_close = lambda a, b, rel=1e-2: False
    res = {}

    def run():
        rs = correctness_reward(None, [c for c, _ in items], task=["math"] * len(items), ref=[r for _, r in items])
        res["out"] = [(r, parsing.extract_answer(c)) for r, (c, _) in zip(rs, items)]
    try:
        if in_thread:
            t = threading.Thread(target=run); t.start(); t.join()
        else:
            run()
    finally:
        (parsing.boxed_contents, verifiers._MV_KW_PARSE, verifiers._MV_KW_VERIFY, verifiers._numeric_close,
         verifiers.VERIFIERS["math"]) = saved
    return res["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config", default="configs/tpu_v6e_7b.yaml")
    ap.add_argument("--out", default=None, help="샘플별 결과 JSONL (기본: <log>.rescored.jsonl)")
    ap.add_argument("--online", action="store_true", help="HF 허브 접속 허용 (기본은 캐시만)")
    a = ap.parse_args()
    out_path = a.out or re.sub(r"\.log$", "", a.log) + ".rescored.jsonl"
    if not a.online:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import yaml
    from tar_rlvr.data.cova import load_cova
    blocks = parse_log(a.log)
    rows = match_refs(blocks, load_cova(yaml.safe_load(open(a.config))["data"]))
    items = [(c, row["ref"]) for (_, comps, _), row in zip(blocks, rows) for c in comps]
    logged = [l for _, _, ls in blocks for l in ls]
    meta = [(bi + 1, j, row["qid"]) for bi, ((_, comps, _), row) in enumerate(zip(blocks, rows)) for j in range(len(comps))]
    print(f"[rescore] {len(blocks)} 롤아웃 블록, {len(items)} completion — 채점 중 (수 분 걸림)", flush=True)

    train = _score_all(items, old_boxed=True, mv_default_timeout=True, legacy=True, in_thread=True)
    mv = _score_all(items, old_boxed=True, mv_default_timeout=False, legacy=True, in_thread=False)
    bx = _score_all(items, old_boxed=False, mv_default_timeout=False, legacy=True, in_thread=False)
    new = _score_all(items, old_boxed=False, mv_default_timeout=False, in_thread=True)
    strict = _score_all(items, old_boxed=False, mv_default_timeout=False, numeric_close=False, in_thread=True)

    n_repro = sum(t[0] == l for t, l in zip(train, logged))
    recs, C = [], collections.Counter()
    for k, ((c, ref), l, (rt, _), (rm, _), (rb, _), (rn, ex), (rs, _), (blk, j, qid)) in enumerate(
            zip(items, logged, train, mv, bx, new, strict, meta)):
        flip = l == 0.0 and rn == 1.0
        cause = None
        if flip:
            cause = "math_verify" if rm == 1.0 else "boxed" if rb == 1.0 else "verifier"
        refd = json.loads(ref)
        suspect = []
        if flip and rs == 0.0:
            suspect.append("numeric_close_only")
        if flip and refd.get("choice") and not (ex and LETTER.match(ex.strip())):
            suspect.append("mc_non_letter")
        recs.append(dict(block=blk, step=(blk + 1) // 2, idx=j, qid=qid, r_log=l, r_train=rt, r_mv=rm, r_boxed=rb, r_new=rn,
                         flip=flip, cause=cause, suspect=suspect, truncated="자 생략]…" in c,
                         extracted=ex, ref=refd, tail=c[-300:]))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    zeros = [r for r in recs if r["r_log"] == 0.0]
    flips = [r for r in recs if r["flip"]]
    sus = [r for r in flips if r["suspect"]]
    lost = [r for r in recs if r["r_log"] == 1.0 and r["r_new"] == 0.0]
    pct = lambda n, d: f"{n}/{d} ({100 * n / max(1, d):.2f}%)"
    print(f"\n학습 당시 채점 재현: {pct(n_repro, len(recs))}  ← 100% 가 아니면 아래 숫자를 믿지 말 것")
    print(f"reward-0 completion: {len(zeros)}")
    print(f"  재채점 시 정답 (0→1): {pct(len(flips), len(zeros))}")
    for cause in ("math_verify", "boxed", "verifier"):
        print(f"    원인 {cause:12s}: {sum(r['cause'] == cause for r in flips)}")
    print(f"    그중 거짓 양성 의심 (suspect): {len(sus)}  "
          f"{dict(collections.Counter(s for r in sus for s in r['suspect']))}")
    print(f"  truncated 샘플 중 0→1: {sum(r['truncated'] for r in flips)}")
    print(f"reward-1 → 재채점 0 (1→0): {len(lost)}")
    print(f"\n샘플별 결과: {out_path}")


if __name__ == "__main__":
    main()
