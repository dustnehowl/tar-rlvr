"""길이 편향 분석: 그림/텍스트 응답 × 정답/오답 의 completion 길이와 sequence-mean 하의 토큰당 gradient 가중치.

출처 (run 에 전 샘플 덤프가 없어서 두 개를 합친다):
  - 디버그 로그 logs/<run>.log (--debug): 샘플마다 reward, advantage (소수 3자리), completion 텍스트
    (시각 토큰 run 은 [IMG S2 ×N] 로 접혀 있고 1200자 넘으면 가운데가 "…[N자 생략]…" 으로 잘림).
  - logs/visual_samples.jsonl: 그림 샘플 (<im_start> 포함) 의 completion 원문 전체 + reward (advantage 없음, 여러 run 이 섞여 있음).
  그림 샘플 → visual_samples 원문을 내용으로 찾아 재토큰화 (정확).
  텍스트 샘플 → 로그 텍스트가 안 잘렸으면 재토큰화 (정확), 잘렸으면 보이는 앞 600 + 뒤 600 자의 토큰/문자 비율로
               생략 부분을 추정 (est=True, 768 상한). 추정 비율을 같이 출력한다.
  재토큰화는 생성 토큰열과 몇 토큰 다를 수 있다. 잘림 판정은 길이 >= max_len - 2.

step 대응: 디버그 로그의 롤아웃 호출 2번 (프롬프트 2개 × G=8) = 학습 1 step. 프로세스가 재시작되면 (같은 로그에
"[data] ... rows" 가 다시 나옴) 뒤 프로세스가 체크포인트에서 이어간 것이므로 뒤 프로세스를 우선하고 마지막 step 에 맞춰
거꾸로 배정한다 (--last-step, 기본: metrics.jsonl 의 마지막 step).

w = |advantage| / length  (sequence-mean-token-mean 에서 한 토큰이 받는 가중치 ∝ |A|/L; 배치 행 수 상수는 생략).

사용: /mnt/work/venv311/bin/python scripts/analysis/length_bias.py --run r06-sketch-bridge [--split 40]
"""
import argparse, json, os, re, statistics, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tar_rlvr.train.debug_log import _s  # 로그에 찍힌 것과 같은 변환 (제어문자 → '·', 시각 토큰 run 접기)

BANNER = "=" * 70
RE_ROLL = re.compile(r"\[2\. 롤아웃\] step (\d+)\s+— (\d+)개 completion")
RE_SAMPLE = re.compile(r"^\s+샘플 (\d+): reward=([+-][\d.]+)\s+adv=([+-][\d.]+)")
RE_SKIP = re.compile(r" …\[(\d+)자 생략\]… ")
IMG_BLOCK = re.compile(r"<im_start>.*?(?:<im_end>|$)", re.S)


def parse_log(path):
    """→ 프로세스별 {롤아웃 호출 번호 (로그의 "step n", 1부터): [ {text, reward, adv}, ... ]}, 프로세스별 최대 호출 번호"""
    procs, maxn, cur, pending = [], [], None, None
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("[data] ") and " rows × " in ln:
            cur = {}
            procs.append(cur)
            maxn.append(0)
            pending = None
        m = RE_ROLL.search(ln)
        if m and cur is not None:
            n, call = int(m.group(2)), int(m.group(1))
            maxn[-1] = max(maxn[-1], call)
            # 다음 줄 BANNER, 그 다음 "--- prompt (공통)" … "#0" … 블록. 다음 BANNER 까지 모은다
            j = i + 2
            body = []
            while j < len(lines) and lines[j] != BANNER and not lines[j].startswith("[data] "):
                body.append(lines[j])
                j += 1
            txt = "\n".join(body)
            parts = re.split(r"\n\n#(\d+)\n--- completion\n", "\n" + txt)
            comps = {}
            for k in range(1, len(parts) - 1, 2):
                comps[int(parts[k])] = parts[k + 1]
            if comps:   # 마지막 completion 뒤에는 banner() 의 "\n" 이 하나 더 붙는다
                last = max(comps)
                comps[last] = comps[last][:-1] if comps[last].endswith("\n") else comps[last]
            pending = (call, [comps[k] for k in range(n)]) if len(comps) == n else None
            i = j
            continue
        if "[3→4. reward 합산 → advantage]" in ln and pending is not None:
            call, pending = pending
            recs, j = [], i + 1
            while j < len(lines) and len(recs) < len(pending):
                ms = RE_SAMPLE.match(lines[j])
                if ms:
                    recs.append((float(ms.group(2)), float(ms.group(3))))
                j += 1
            if len(recs) == len(pending):
                cur[call] = [dict(text=t, reward=r, adv=a) for t, (r, a) in zip(pending, recs)]
            pending = None
            i = j
            continue
        i += 1
    return procs, maxn


def assign_steps(procs, maxn, last_step, cps):
    """첫 프로세스는 step 0 부터. 마지막 프로세스 (재시작=체크포인트에서 이어감) 는 마지막 호출이 last_step 이 되게.
    겹치는 step 은 뒤 프로세스 우선. 중간 프로세스 (재시작 2번 이상) 는 위치를 알 수 없어 버린다 (경고).
    롤아웃 호출 cps 개가 모두 파싱된 step 만 쓴다."""
    starts = [0] * len(procs)
    if len(procs) > 1:
        starts[-1] = last_step - (maxn[-1] + cps - 1) // cps + 1
        if len(procs) > 2:
            print(f"[warn] 프로세스 {len(procs)}개: 중간 {len(procs) - 2}개는 버림", file=sys.stderr)
    out, dropped = {}, 0
    for p, (calls, s0) in enumerate(zip(procs, starts)):
        if 0 < p < len(procs) - 1:
            continue
        for k in range((maxn[p] + cps - 1) // cps):
            ids = [k * cps + c + 1 for c in range(cps)]
            if not all(i in calls for i in ids):
                dropped += 1
                continue
            s = s0 + k
            if 0 <= s <= last_step and (p == len(procs) - 1 or s < starts[-1]):
                out[s] = [smp for i in ids for smp in calls[i]]
    print(f"[log] step 시작 {starts}, 호출이 다 안 잡혀 버린 step {dropped}", file=sys.stderr)
    return dict(sorted(out.items()))


def key_of(shown):
    shown = shown.rstrip()
    return (shown[:300], shown[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", default=None, help="기본 logs/<run>.log")
    ap.add_argument("--visual-samples", default="logs/visual_samples.jsonl")
    ap.add_argument("--metrics", default=None, help="기본 logs/runs/<run>/metrics.jsonl")
    ap.add_argument("--last-step", type=int, default=None)
    ap.add_argument("--tokenizer", default="/mnt/work/tar7b_lm")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--bridge", default="\nNow, reasoning step by step with the help of the sketch:\n\nStep 1:")
    ap.add_argument("--calls-per-step", type=int, default=2)
    ap.add_argument("--split", type=int, default=40, help="Table 3: steps 0..split-1 / split..end")
    ap.add_argument("--dump", default=None, help="샘플 단위 결과 jsonl (선택)")
    a = ap.parse_args()
    log = a.log or f"logs/{a.run}.log"
    met = a.metrics or f"logs/runs/{a.run}/metrics.jsonl"
    if a.last_step is None:
        a.last_step = max(json.loads(l)["step"] for l in open(met) if "learning_signal/reward_mean" in l)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0
    n_bridge = ntok(a.bridge)

    mtr = {}   # step → length/truncation_rate (토큰 id 기준: EOS 없이 n >= max_len-1) — 텔레메트리의 정확한 값
    for l in open(met):
        d = json.loads(l)
        if "length/truncation_rate" in d:
            mtr[d["step"]] = d["length/truncation_rate"]
    procs, maxn = parse_log(log)
    steps = assign_steps(procs, maxn, a.last_step, a.calls_per_step)
    print(f"[log] 프로세스 {len(procs)}개, 호출 {[len(p) for p in procs]} → step {min(steps)}–{max(steps)} ({len(steps)} steps, "
          f"{sum(len(v) for v in steps.values())} 샘플); bridge {n_bridge} 토큰", file=sys.stderr)

    vis = {}
    for l in open(a.visual_samples, encoding="utf-8"):
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        c = d["completion"]
        vis.setdefault((key_of(_s(c)), round(float(d["reward"]), 3)), c)

    rows, miss = [], 0
    for s, smps in steps.items():
        for smp in smps:
            t = smp["text"]
            img = "[IMG" in t or "<im_start>" in t
            r = dict(step=s, reward=smp["reward"], adv=smp["adv"], image=img, est=False)
            if img:
                full = vis.get((key_of(t), round(smp["reward"], 3)))
                if full is None:
                    miss += 1
                    continue
                first = full.find("<im_start>")
                pre = full[:first]
                blocks = IMG_BLOCK.findall(full[first:])
                rest = IMG_BLOCK.sub("", full[first:])
                has_bridge = a.bridge in full
                post = rest.replace(a.bridge, "", 1) if has_bridge else rest
                r.update(pre=ntok(pre), visual=sum(ntok(b) for b in blocks), bridge=n_bridge if has_bridge else 0, post=ntok(post),
                         n_images=len(blocks))
                r["length"] = r["pre"] + r["visual"] + r["bridge"] + r["post"]
            else:
                m = RE_SKIP.search(t)
                if m:
                    head, tail = t[:m.start()], t[m.end():]
                    seen = ntok(head) + ntok(tail)
                    est = seen + int(m.group(1)) * seen / max(1, len(head) + len(tail))
                    r.update(length=int(min(a.max_len, round(est))), est=True)
                else:
                    r["length"] = ntok(t)
            r["trunc"] = r["length"] >= a.max_len - 2
            r["w"] = abs(r["adv"]) / max(1, r["length"])
            rows.append(r)
    print(f"[join] 그림 샘플 중 visual_samples 에서 못 찾음: {miss} (제외)", file=sys.stderr)
    if a.dump:
        with open(a.dump, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def report(rs, title):
        print(f"\n### {title}\n")
        n_est = sum(r["est"] for r in rs)
        print(f"샘플 {len(rs)} (그림 {sum(r['image'] for r in rs)}, 텍스트 {sum(not r['image'] for r in rs)}); "
              f"텍스트 길이 추정 (로그에서 가운데 잘림) {n_est}\n")
        cell = lambda img, ok: [r for r in rs if r["image"] == img and (r["reward"] >= 0.5) == ok]
        print("Table 1 — completion length (tokens): median / mean (n)\n")
        print("| | correct | incorrect |\n|---|---|---|")
        for img, nm in ((True, "image"), (False, "text")):
            cs = []
            for ok in (True, False):
                L = [r["length"] for r in cell(img, ok)]
                cs.append(f"{statistics.median(L):.0f} / {np.mean(L):.1f} ({len(L)})" if L else "— (0)")
            print(f"| {nm} | {cs[0]} | {cs[1]} |")
        print("\nimage responses — segment means (tokens), mean (median):\n")
        print("| | n | text before image | visual block(s) | bridge | text after bridge | images/resp | with bridge |\n|---|---|---|---|---|---|---|---|")
        for ok, nm in ((True, "correct"), (False, "incorrect"), (None, "all")):
            c = [r for r in rs if r["image"] and (ok is None or (r["reward"] >= 0.5) == ok)]
            if not c:
                continue
            f = lambda k: f"{np.mean([r[k] for r in c]):.1f} ({statistics.median([r[k] for r in c]):.0f})"
            print(f"| {nm} | {len(c)} | {f('pre')} | {f('visual')} | {f('bridge')} | {f('post')} | "
                  f"{np.mean([r['n_images'] for r in c]):.2f} | {np.mean([r['bridge'] > 0 for r in c]):.2f} |")
        print("\nTable 2 — mean w = |adv| / length  (×1e-3), n; and mean |adv|\n")
        print("| | correct | incorrect |\n|---|---|---|")
        W = {}
        for img, nm in ((True, "image"), (False, "text")):
            cs = []
            for ok in (True, False):
                c = cell(img, ok)
                W[(img, ok)] = np.mean([r["w"] for r in c]) if c else float("nan")
                cs.append(f"{1e3 * W[(img, ok)]:.3f} (n={len(c)}, |adv| {np.mean([abs(r['adv']) for r in c]):.3f})" if c else "—")
            print(f"| {nm} | {cs[0]} | {cs[1]} |")
        # 오답 중 adv=0 (그룹 전원 오답) 은 gradient 0 → 실제 벌점을 받은 샘플만 따로
        cnz = lambda img: [r for r in cell(img, False) if r["adv"] != 0]
        wt, wi = np.mean([r["w"] for r in cnz(False)] or [np.nan]), np.mean([r["w"] for r in cnz(True)] or [np.nan])
        print(f"\nw_text_incorrect / w_image_incorrect = {W[(False, False)] / W[(True, False)]:.2f} (all incorrect); "
              f"{wt / wi:.2f} (incorrect with adv≠0: text n={len(cnz(False))}, image n={len(cnz(True))})")
        print("\ntruncated at max length (len >= {}):\n".format(a.max_len - 2))
        print("| | n truncated | of | rate |\n|---|---|---|---|")
        for img, nm in ((True, "image"), (False, "text")):
            c = [r for r in rs if r["image"] == img]
            k = sum(r["trunc"] for r in c)
            print(f"| {nm} | {k} | {len(c)} | {k / max(1, len(c)):.3f} |")
        ss = sorted({r["step"] for r in rs if r["step"] in mtr})
        per = {st: sum(1 for r in rs if r["step"] == st) for st in ss}
        tot = sum(round(mtr[st] * per[st]) for st in ss)
        ki = sum(r["trunc"] for r in rs if r["image"] and r["step"] in mtr)
        nt = sum(1 for r in rs if not r["image"] and r["step"] in mtr)
        print(f"\nmetrics.jsonl length/truncation_rate (토큰 id, EOS 없음 & n >= {a.max_len - 1}): 합계 {tot} / {sum(per.values())}; "
              f"text = 합계 − image({ki}) = {tot - ki} / {nt} ({(tot - ki) / max(1, nt):.3f})")

    lo, hi = min(steps), max(steps)
    report(rows, f"{a.run}: all steps {lo}–{hi}")
    report([r for r in rows if r["step"] < a.split], f"{a.run}: steps {lo}–{a.split - 1}")
    report([r for r in rows if r["step"] >= a.split], f"{a.run}: steps {a.split}–{hi}")


if __name__ == "__main__":
    main()
