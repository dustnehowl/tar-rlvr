"""Stage 1 결과 분석: logs/stage1/<run>/records.jsonl + prompts.jsonl → 표 (markdown) + summary.json

- 정답률: greedy (Wilson 95% CI), sampled pass@1 / pass@8 (프롬프트 단위 bootstrap 95% CI).
- 짝 비교: greedy 는 McNemar (exact), sampled 는 프롬프트 단위 paired bootstrap (pass@1, pass@8).
- 부분집합: leaky 제외(main) / 전체, 객관식 vs 서술형, <im_start> 를 낸 프롬프트 vs 안 낸 프롬프트.
  greedy: prefix 가 <im_start> 를 냈는지 (그림 세 조건에서 공통). sampled: 8 샘플 중 하나라도 냈으면 'emitted'.
  sampled 는 추가로 '<im_start> 를 낸 샘플만' 의 샘플 단위 비교도 낸다 (그림 세 조건은 prefix 를 공유해 정확히 짝).
- <im_start> 위치: prefix 토큰 수 (completion 시작부터 몇 토큰 뒤), 최종 completion 길이 대비 상대 위치.
"""
import argparse, collections, json, math, os, random
import numpy as np

ORDER = ["text_only", "model_drawn", "model_drawn_S0", "oracle", "oracle_S2", "oracle_S0", "corrupt", "corrupt_S0"]
ALL_PAIRS = [("oracle", "text_only"), ("oracle", "corrupt"), ("oracle", "model_drawn"),
             ("oracle_S2", "text_only"), ("oracle_S2", "corrupt"), ("oracle_S2", "model_drawn"),
             ("oracle_S0", "text_only"), ("oracle_S0", "corrupt"), ("oracle_S0", "oracle_S2"), ("oracle_S0", "model_drawn_S0"),
             ("model_drawn_S0", "model_drawn"), ("model_drawn", "text_only"), ("model_drawn_S0", "text_only"),
             ("corrupt", "text_only"),
             ("oracle_S0", "corrupt_S0"), ("corrupt_S0", "text_only")]           # Stage 1b
CONDS, PAIRS = list(ORDER), list(ALL_PAIRS)      # main() 에서 records 에 있는 조건으로 좁힌다


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    """a, b: bool 배열 (짝). exact 양측 p."""
    b10 = int(np.sum(a & ~b)); b01 = int(np.sum(~a & b)); n = b10 + b01
    if n == 0:
        return b10, b01, 1.0
    k = min(b10, b01)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b10, b01, min(1.0, p)


def boot(x, B=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_ci(m, lo, hi, pct=True):
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:+.3f}")
    return f"{f(m)} [{f(lo)}, {f(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/stage1/base")
    ap.add_argument("--prompts", default="logs/stage1/prompts.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = {r["qid"]: r for r in map(json.loads, open(a.prompts))}
    R = [json.loads(l) for l in open(f"{a.run}/records.jsonl")]
    global CONDS, PAIRS
    present = {r["cond"] for r in R}
    CONDS = [c for c in ORDER if c in present]
    PAIRS = [(x, y) for x, y in ALL_PAIRS if x in present and y in present]
    IMG = [c for c in CONDS if c != "text_only"]
    md, summary = [], {}
    p = md.append

    n_leaky = sum(r["leaky"] for r in P.values())
    p(f"# Stage 1 — oracle on demand ({a.run})\n")
    p(f"프롬프트 {len(P)} (held-out {sum(r['split'] == 'heldout' for r in P.values())}), "
      f"객관식 {sum(r['is_mc'] for r in P.values())}, leaky (휴리스틱) {n_leaky} "
      f"→ main = leaky 제외 {len(P) - n_leaky}.\n")

    for kind in ("greedy", "sampled"):
        rk = [r for r in R if r["kind"] == kind]
        if not rk:
            continue
        # by[cond][qid] = [correct per sample]
        by = {c: collections.defaultdict(dict) for c in CONDS}
        recs = {c: collections.defaultdict(dict) for c in CONDS}
        for r in rk:
            by[r["cond"]][r["qid"]][r["sample"]] = r["correct"]
            recs[r["cond"]][r["qid"]][r["sample"]] = r
        qids = sorted(set.intersection(*[set(by[c]) for c in CONDS]))
        G = max(len(v) for v in by[IMG[0]].values())
        emitted_s = {q: [recs[IMG[0]][q][j]["emitted"] for j in range(G)] for q in qids}
        emit_q = {q: any(v) for q, v in emitted_s.items()}

        subsets = collections.OrderedDict()
        main = [q for q in qids if not P[q]["leaky"]]
        subsets["main (non-leaky)"] = main
        subsets["main · free-form"] = [q for q in main if not P[q]["is_mc"]]
        subsets["main · MC"] = [q for q in main if P[q]["is_mc"]]
        subsets["main · emitted"] = [q for q in main if emit_q[q]]
        subsets["main · emitted · free-form"] = [q for q in main if emit_q[q] and not P[q]["is_mc"]]
        subsets["main · emitted · MC"] = [q for q in main if emit_q[q] and P[q]["is_mc"]]
        subsets["main · not emitted"] = [q for q in main if not emit_q[q]]
        subsets["main · not emitted · free-form"] = [q for q in main if not emit_q[q] and not P[q]["is_mc"]]
        subsets["leaky only"] = [q for q in qids if P[q]["leaky"]]
        subsets["all 300"] = qids
        if any("solvable" in P[q] for q in qids):                  # Stage 1b maze
            subsets["main · solvable"] = [q for q in main if P[q].get("solvable")]
            subsets["main · UNSOLVABLE"] = [q for q in main if P[q].get("solvable") is False]
            subsets["main · emitted · solvable"] = [q for q in main if emit_q[q] and P[q].get("solvable")]
        for k in [k for k, v in subsets.items() if not v]:          # 빈 부분집합 (비수학 태스크의 MC 등) 제외
            del subsets[k]

        p(f"\n## {kind} ({'1 샘플' if kind == 'greedy' else f'{G} 샘플, T=0.8'})\n")
        p(f"<im_start> 를 낸 프롬프트: {sum(emit_q.values())}/{len(qids)}"
          + ("" if kind == "greedy" else f" (샘플 기준 {sum(sum(v) for v in emitted_s.values())}/{len(qids) * G})") + "\n")
        # ---- 조건별 정답률
        metric = "greedy acc" if kind == "greedy" else "pass@1 / pass@8"
        p(f"\n### 조건별 {metric} (%, 95% CI)\n")
        p("| subset | n | " + " | ".join(CONDS) + " |")
        p("|---|---|" + "---|" * len(CONDS))
        S = summary.setdefault(kind, {})
        for name, qs in subsets.items():
            cells = []
            for c in CONDS:
                arr = np.array([[by[c][q][j] for j in range(G)] for q in qs], bool) if qs else np.zeros((0, G), bool)
                if kind == "greedy":
                    k = int(arr.sum()); lo, hi = wilson(k, len(qs))
                    cells.append(fmt_ci(k / max(1, len(qs)), lo, hi))
                    S.setdefault(name, {})[c] = {"acc": k / max(1, len(qs)), "n": len(qs)}
                else:
                    p1 = arr.mean(1) if len(qs) else np.zeros(0); p8 = arr.any(1).astype(float) if len(qs) else np.zeros(0)
                    cells.append(f"{100 * p1.mean():.1f} / {100 * p8.mean():.1f}" if len(qs) else "–")
                    S.setdefault(name, {})[c] = {"pass1": float(p1.mean()) if len(qs) else None,
                                                 "passk": float(p8.mean()) if len(qs) else None, "n": len(qs)}
            p(f"| {name} | {len(qs)} | " + " | ".join(cells) + " |")
        # ---- 짝 비교
        p(f"\n### 짝 비교 (A − B)\n")
        if kind == "greedy":
            p("| subset | n | A − B | diff (pp) | A만 맞음 / B만 맞음 | McNemar p |")
            p("|---|---|---|---|---|---|")
        else:
            p("| subset | n | A − B | Δpass@1 (pp) [95% CI] | Δpass@8 (pp) [95% CI] |")
            p("|---|---|---|---|---|")
        for name in ("main (non-leaky)", "main · free-form", "main · MC", "main · emitted", "main · emitted · free-form",
                     "main · emitted · MC", "main · not emitted", "main · not emitted · free-form",
                     "main · solvable", "main · UNSOLVABLE"):
            if name not in subsets:
                continue
            qs = subsets[name]
            if not qs:
                continue
            for A, Bc in PAIRS:
                xa = np.array([[by[A][q][j] for j in range(G)] for q in qs], bool)
                xb = np.array([[by[Bc][q][j] for j in range(G)] for q in qs], bool)
                key = f"{A}-{Bc}"
                if kind == "greedy":
                    b10, b01, pv = mcnemar(xa[:, 0], xb[:, 0])
                    d = (xa[:, 0].mean() - xb[:, 0].mean())
                    p(f"| {name} | {len(qs)} | {A} − {Bc} | {100 * d:+.1f} | {b10} / {b01} | {pv:.3g} |")
                    S[name].setdefault("pairs", {})[key] = {"diff": float(d), "b10": b10, "b01": b01, "p": pv}
                else:
                    d1 = xa.mean(1) - xb.mean(1); d8 = xa.any(1).astype(float) - xb.any(1).astype(float)
                    c1, c8 = boot(d1), boot(d8)
                    p(f"| {name} | {len(qs)} | {A} − {Bc} | {100 * d1.mean():+.1f} [{100 * c1[0]:+.1f}, {100 * c1[1]:+.1f}] "
                      f"| {100 * d8.mean():+.1f} [{100 * c8[0]:+.1f}, {100 * c8[1]:+.1f}] |")
                    S[name].setdefault("pairs", {})[key] = {"d_pass1": float(d1.mean()), "ci_pass1": c1,
                                                            "d_passk": float(d8.mean()), "ci_passk": c8}
        if kind == "sampled":
            # 샘플 단위: <im_start> 를 낸 샘플만 (그림 세 조건은 prefix 공유 → 정확한 짝). text_only 는 짝이 없어 제외.
            p("\n### <im_start> 를 낸 샘플만 (샘플 단위, 그림 세 조건; prefix 공유)\n")
            OR = "oracle_S2" if "oracle_S2" in IMG else "oracle"
            spairs = [(x, y) for x, y in PAIRS if x in IMG and y in IMG]
            p("| subset | n 샘플 | " + " | ".join(IMG) + " | " + " | ".join(f"{x} − {y} [CI]" for x, y in spairs) + " |")
            p("|---|---|" + "---|" * (len(IMG) + len(spairs)))
            for name in [n for n in ("main (non-leaky)", "main · free-form", "main · MC", "main · solvable") if n in subsets]:
                qs = subsets[name]
                groups = []
                for q in qs:
                    js = [j for j in range(G) if emitted_s[q][j]]
                    if js:
                        groups.append([[by[c][q][j] for c in IMG] for j in js])
                flat = np.array([s for g in groups for s in g], float)
                if len(flat) == 0:
                    continue
                # 프롬프트 클러스터 bootstrap (같은 프롬프트의 샘플은 독립 아님)
                rng = np.random.default_rng(0)
                ix = [(IMG.index(x), IMG.index(y)) for x, y in spairs]
                ds = [[] for _ in ix]
                for _ in range(5000):
                    pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
                    f = np.array([s for g in pick for s in g], float).mean(0)
                    for k, (u, v) in enumerate(ix):
                        ds[k].append(f[u] - f[v])
                m = flat.mean(0)
                p(f"| {name} | {len(flat)} ({len(groups)} 프롬프트) | " + " | ".join(f"{100 * v:.1f}" for v in m) + " | "
                  + " | ".join(f"{100 * (m[u] - m[v]):+.1f} [{100 * np.percentile(d, 2.5):+.1f}, {100 * np.percentile(d, 97.5):+.1f}]"
                               for (u, v), d in zip(ix, ds)) + " |")
        # ---- <im_start> 위치
        pos = [recs[IMG[0]][q][j] for q in qids for j in range(G) if recs[IMG[0]][q][j]["emitted"]]
        if pos:
            fp = np.array([r["first_pos"] for r in pos])
            md_len = np.array([recs["model_drawn"][r["qid"]][r["sample"]]["n_tokens"] for r in pos])
            rel = fp / np.maximum(1, md_len)
            p(f"\n### <im_start> 위치 ({len(pos)} 샘플)\n")
            p(f"- completion 시작부터 토큰 수: 중앙값 {np.median(fp):.0f}, 사분위 [{np.percentile(fp, 25):.0f}, {np.percentile(fp, 75):.0f}], "
              f"최소 {fp.min()}, 최대 {fp.max()}")
            buckets = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100), (101, 200), (201, 768)]
            p("- 분포: " + ", ".join(f"{lo}–{hi}: {int(((fp >= lo) & (fp <= hi)).sum())}" for lo, hi in buckets))
            p(f"- model_drawn 최종 길이 대비 상대 위치: 중앙값 {np.median(rel):.2f}, 사분위 [{np.percentile(rel, 25):.2f}, {np.percentile(rel, 75):.2f}]")
            S["im_start_pos"] = {"n": len(pos), "median": float(np.median(fp)), "q25": float(np.percentile(fp, 25)),
                                 "q75": float(np.percentile(fp, 75)), "frac_0": float((fp == 0).mean()),
                                 "frac_le5": float((fp <= 5).mean()), "rel_median": float(np.median(rel))}
        # ---- 행동 지표
        p(f"\n### 행동 지표 (main)\n")
        p("| cond | 답 도달률 | 잘림률 | 평균 그림 수 | 그림 2장 이상 | 토큰 중앙값 |")
        p("|---|---|---|---|---|---|")
        for c in CONDS:
            rr = [recs[c][q][j] for q in subsets["main (non-leaky)"] for j in range(G)]
            p(f"| {c} | {100 * np.mean([r['has_answer'] for r in rr]):.1f} | {100 * np.mean([r['truncated'] for r in rr]):.1f} "
              f"| {np.mean([r['n_images'] for r in rr]):.2f} | {100 * np.mean([r['n_images'] >= 2 for r in rr]):.1f} "
              f"| {np.median([r['n_tokens'] for r in rr]):.0f} |")
    out = a.out or f"{a.run}/report.md"
    open(out, "w").write("\n".join(md) + "\n")
    spath = f"{a.run}/summary.json" if a.out is None else os.path.splitext(out)[0] + "_summary.json"
    json.dump(summary, open(spath, "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
