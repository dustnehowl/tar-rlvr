"""텔레메트리(또는 옛 --debug 로그)를 W&B 로 올리는 사이드카. 학습 프로세스와 분리된 venv 에서 돈다.

사용 (/mnt/work/wandb-venv — 학습 venv 의존성 건드리지 않음):
  # 새 run: 학습이 logs/runs/<run>/ 에 쓰는 것을 따라가며 업로드
  /mnt/work/wandb-venv/bin/python scripts/wandb_sync.py logs/runs/<run> --entity dustnehowl --follow
  # 옛 run: --debug 콘솔 로그에서 뽑을 수 있는 지표만 (entropy/KL/grad norm/토큰 길이/eval 은 없음)
  /mnt/work/wandb-venv/bin/python scripts/wandb_sync.py logs/tpu_run3.log --name run3 --entity dustnehowl

모든 지표를 원값 + 10 step 이동평균(<이름>_ma10) 으로 올린다 (eval/ 은 원값만). x 축은 train_step.
learning_signal/samples_seen = 누적 completion 수.
samples/ 의 completion 표, 그림 디코딩(visual/decoded) 은 학습 venv 의 scripts/decode_visual.py 를 하위 프로세스로 돌린다.
같은 --name 으로 다시 실행하면 이어서 올린다.
"""
from __future__ import annotations
import argparse, collections, glob, json, math, os, re, statistics, subprocess, sys, time

import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tar_rlvr.rewards.parsing import ANSWER_RE, boxed_contents, extract_answer   # noqa: E402  (순수 파이썬)
from tar_rlvr.rewards.verifiers import _canon, _norm                              # noqa: E402

TRAIN_PY = "/mnt/work/venv311/bin/python"
MA = 10
NO_SMOOTH = ("eval/",)
_IMG_RUN = re.compile(r"(<S\d>)?((?:<I\d+>)+)")


def collapse(text: str) -> str:
    return _IMG_RUN.sub(lambda m: f"[IMG{(' ' + m.group(1)) if m.group(1) else ''} ×{m.group(2).count('<I')}]", text)


class Uploader:
    def __init__(self, run, per_step: int):
        self.run, self.per_step = run, per_step
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=MA))
        self.last = {}
        run.define_metric("train_step")
        run.define_metric("*", step_metric="train_step")

    def emit(self, step: int, metrics: dict, extra: dict | None = None):
        out = {"train_step": step, "learning_signal/samples_seen": (step + 1) * self.per_step}
        for k, v in sorted(metrics.items()):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out[k] = v
            if not k.startswith(NO_SMOOTH):
                h = self.hist[k]
                h.append(v)
                out[f"{k}_ma{MA}"] = sum(h) / len(h)
        self.last.update(out)
        out.update(extra or {})
        self.run.log(out)

    def flush_summary(self):
        """define_metric(step_metric=train_step) 을 쓰면 이 W&B 서버에선 summary 가 비어 run 비교 표가 안 채워진다 → 직접 기록."""
        if self.last:
            self.run.summary.update(self.last)


# ------------------------------------------------------------------------------------------ 텔레메트리 디렉토리
class RunDirSource:
    def __init__(self, path: str, up: Uploader, done_step: int, decode: bool):
        self.path, self.up, self.done, self.decode = path, up, done_step, decode
        self.offset = 0
        self.buf = collections.defaultdict(dict)
        self.jobs = {}                       # step -> (Popen, out_dir)
        self.tabled = set()

    def poll(self, final: bool):
        f = os.path.join(self.path, "metrics.jsonl")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                fh.seek(self.offset)
                for line in fh:
                    if not line.endswith("\n"):
                        break                # 쓰는 중인 줄
                    self.offset += len(line.encode("utf-8"))
                    r = json.loads(line)
                    s = r.pop("step"); r.pop("t", None)
                    if s >= 0:
                        self.buf[s].update(r)
        # 학습 지표(actor)는 다음 step 에 도착하므로 2 step 뒤처져서 확정. 파일이 10분간 안 바뀌면(학습 끝) 전부 확정.
        idle = os.path.exists(f) and time.time() - os.path.getmtime(f) > 600
        ready_until = max(self.buf, default=-1) - (0 if final or idle else 2)
        for s in sorted(k for k in self.buf if k <= ready_until):
            if s > self.done:
                self.up.emit(s, self.buf[s])
                self.done = s
            del self.buf[s]
        self._samples(final)

    def _samples(self, final: bool):
        for fp in sorted(glob.glob(os.path.join(self.path, "samples", "step_*.jsonl"))):
            s = int(re.search(r"step_(\d+)", fp).group(1))
            if s in self.tabled or (s > self.done and not final):
                continue
            rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
            t = wandb.Table(columns=["step", "reward", "adv", "has_image", "forced", "length", "extracted", "gold", "completion", "prompt"])
            for r in rows:
                gold = r.get("ref", {}).get("answer") if isinstance(r.get("ref"), dict) else None
                t.add_data(s, r["reward"], r.get("adv"), r["has_image"], r.get("forced", False), r["length"], r.get("extracted"), gold,
                           collapse(r["completion"]), r["prompt"][-800:])
            self.up.run.log({"train_step": s, "samples/completions": t})
            self.tabled.add(s)
            if self.decode and any("<im_end>" in r["completion"] for r in rows) and s not in self.jobs:
                out = os.path.join(self.path, "vis", f"step_{s:06d}")
                if glob.glob(os.path.join(out, "*.png")):
                    self.jobs[s] = (None, out)
                else:
                    os.makedirs(out, exist_ok=True)
                    log = open(os.path.join(out, "decode.log"), "w")
                    p = subprocess.Popen(["nice", "-n", "15", TRAIN_PY, os.path.join(ROOT, "scripts/decode_visual.py"),
                                          "--jsonl", fp, "--out", out, "--n", "4", "--any-reward"],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    self.jobs[s] = (p, out)
        for s, (p, out) in list(self.jobs.items()):
            if p is not None and p.poll() is None:
                if final:
                    p.wait()
                else:
                    continue
            pngs = sorted(glob.glob(os.path.join(out, "*.png")))
            if pngs:
                self.up.run.log({"train_step": s, "visual/decoded": [wandb.Image(x, caption=os.path.basename(x)) for x in pngs]})
            else:
                print(f"[wandb_sync] step {s} 그림 디코딩 결과 없음 — {out}/decode.log 확인", flush=True)
            del self.jobs[s]

    def busy(self):
        return bool(self.jobs)


# ------------------------------------------------------------------------------------------ 옛 --debug 로그
HDR = re.compile(r"\[2\. 롤아웃\] step (\d+)")
CUT = re.compile(r"…\[(\d+)자 생략\]…")
IMG = re.compile(r"\[IMG(?: <S\d>)? ×(\d+)\]")
BOXED_ONLY = re.compile(r"^\s*\$*\s*\\boxed\{.*\}\s*\$*\s*\.?\s*$", re.S)


def parse_log(txt):
    parts = HDR.split(txt)[1:]
    out = []
    for no, body in zip(parts[::2], parts[1::2]):
        roll, sep, rew = body.partition("[3→4. reward 합산")
        if not sep:
            break
        comps = [c.split("\n=====")[0].rstrip("\n") for c in re.split(r"\n#\d+\n--- completion\n", roll)[1:]]
        ra = [(float(a), float(b)) for a, b in re.findall(r"샘플 \d+: reward=([+-]\d+\.\d+)\s+adv=([+-]\d+\.\d+)", rew)]
        if not comps or len(ra) != len(comps):
            break
        out.append((int(no), comps, [x for x, _ in ra], [y for _, y in ra]))
    return out


def true_len(c):
    return len(CUT.sub("", c)) + sum(int(n) for n in CUT.findall(c))


def _mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(x) / len(x) if x else None


def _med(x):
    return statistics.median(x) if x else None


def log_metrics(groups):
    """groups: [(comps, rewards, advs)] (그룹 = 한 문제의 G개) → 로그에서 계산 가능한 지표."""
    C, R, A, OK, IMGF, ANS, TAG = [], [], [], [], [], [], []
    learn, z, o, pk, maj, paired = [], [], [], [], [], []
    for comps, rs, advs in groups:
        ok = [r >= 1.0 for r in rs]
        im = ["[IMG" in c for c in comps]
        ans = [extract_answer(c) for c in comps]
        learn.append(float(max(rs) - min(rs) > 1e-8)); z.append(float(not any(ok))); o.append(float(all(ok))); pk.append(float(any(ok)))
        votes = collections.Counter(_canon(_norm(a)) for a in ans if a)
        if votes:
            top = votes.most_common(1)[0][0]
            maj.append(float(any(k for k, a in zip(ok, ans) if a and _canon(_norm(a)) == top)))
        else:
            maj.append(0.0)
        if any(im) and not all(im):
            ov = [k for k, i in zip(ok, im) if i]; ot = [k for k, i in zip(ok, im) if not i]
            paired.append(sum(ov) / len(ov) - sum(ot) / len(ot))
        C += comps; R += rs; A += advs; OK += ok; IMGF += im; ANS += ans
        TAG += [bool(ANSWER_RE.search(c) or boxed_contents(c)) for c in comps]
    n = len(C)
    L = [true_len(c) for c in C]
    Lt = [true_len(IMG.sub("", c)) for c in C]
    return {
        "learning_signal/reward_mean": sum(R) / n, "learning_signal/acc": sum(OK) / n,
        "learning_signal/learnable_frac": _mean(learn), "learning_signal/all_zero_frac": _mean(z),
        "learning_signal/all_one_frac": _mean(o), "learning_signal/adv_abs_mean": sum(abs(a) for a in A) / n,
        "learning_signal/pass_at_1": sum(OK) / n, f"learning_signal/pass_at_{len(groups[0][0])}": _mean(pk),
        "learning_signal/maj_vote_acc": _mean(maj),
        "visual/img_frac": sum(IMGF) / n,
        "visual/acc_vis": _mean([float(k) for k, i in zip(OK, IMGF) if i]),
        "visual/acc_txt": _mean([float(k) for k, i in zip(OK, IMGF) if not i]),
        "visual/paired_acc_diff": _mean(paired), "visual/paired_group_frac": len(paired) / len(groups),
        "visual/adv_img_mean": _mean([a for a, i in zip(A, IMGF) if i]),
        "visual/images_per_completion": sum(len(IMG.findall(c)) for c in C) / n,
        "visual/visual_tokens_per_completion": sum(int(k) for c in C for k in IMG.findall(c)) / n,
        "length/chars_median": _med(L), "length/text_chars_median": _med(Lt),
        "length/chars_median_correct": _med([l for l, k in zip(L, OK) if k]),
        "length/chars_median_incorrect": _med([l for l, k in zip(L, OK) if not k]),
        "length/boxed_only_frac": sum(bool(BOXED_ONLY.match(c)) for c in C) / n,
        "verifier/zero_parse_fail": sum((not k) and a is None for k, a in zip(OK, ANS)) / n,
        "verifier/zero_format_fail": sum((not k) and a is not None and not t for k, a, t in zip(OK, ANS, TAG)) / n,
        "verifier/zero_wrong_answer": sum((not k) and a is not None and t for k, a, t in zip(OK, ANS, TAG)) / n,
    }


class LogSource:
    def __init__(self, path, up: Uploader, done_step: int, bps: int, sample_every: int):
        self.path, self.up, self.done, self.bps, self.sample_every = path, up, done_step, bps, sample_every

    def poll(self, final: bool):
        blocks = parse_log(open(self.path, encoding="utf-8", errors="replace").read())
        for s in range(self.done + 1, len(blocks) // self.bps):        # 0부터: 텔레메트리와 같은 기준
            grp = blocks[s * self.bps:(s + 1) * self.bps]
            extra = {}
            if self.sample_every and s % self.sample_every == 0:
                t = wandb.Table(columns=["step", "block", "reward", "adv", "has_image", "length", "extracted", "completion"])
                for no, cs, rs, advs in grp:
                    for c, r, a in zip(cs, rs, advs):
                        t.add_data(s, no, r, a, "[IMG" in c, true_len(c), extract_answer(c), c)
                extra["samples/completions"] = t
            self.up.emit(s, log_metrics([(cs, rs, advs) for _, cs, rs, advs in grp]), extra)
            self.done = s

    def busy(self):
        return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def auto_tags(cfg: dict) -> list[str]:
    """run 을 W&B 에서 걸러 보기 위한 태그: 지시문 / 그림 처리 방식."""
    if not cfg:
        return []
    g = cfg.get("grpo", {})
    fi, cont = g.get("forced_image") or {}, g.get("continuation") or {}
    image = (f"forced{fi.get('per_group')}of{g.get('num_generations')}" if fi.get("enabled")
             else "bridge" if cont.get("enabled", True) else "free")
    return [f"prompt:{cfg.get('data', {}).get('instruction', 'sketch')}", f"image:{image}", f"beta:{g.get('beta')}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="logs/runs/<run> 디렉토리 또는 옛 --debug 로그 파일")
    ap.add_argument("--name", default=None, help="W&B run 이름이자 id (기본: 디렉토리 이름)")
    ap.add_argument("--project", default="tar-rlvr")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--no-decode", action="store_true", help="그림 디코딩 끄기")
    ap.add_argument("--sample-every", type=int, default=50, help="(옛 로그) completion 표 주기")
    ap.add_argument("--until-pid", type=int, default=None, help="--follow 중 이 PID(학습)가 끝나면 마지막까지 올리고 종료")
    ap.add_argument("--tags", default="", help="쉼표 구분 W&B 태그 (기본: config 에서 자동)")
    a = ap.parse_args()

    is_dir = os.path.isdir(a.source)
    name = a.name or os.path.basename(a.source.rstrip("/"))
    cfg = {}
    if is_dir and os.path.exists(os.path.join(a.source, "config.json")):
        cfg = json.load(open(os.path.join(a.source, "config.json")))
    g = cfg.get("grpo", {})
    per_step = g.get("batch_size", 2) * g.get("num_generations", 8)
    run = wandb.init(entity=a.entity, project=a.project, name=name, id=name, resume="allow",
                     config={"source": a.source, **cfg}, tags=[t for t in a.tags.split(",") if t] or auto_tags(cfg),
                     settings=wandb.Settings(silent=True))
    done = run.summary.get("train_step")
    if done is None and run.resumed:            # 서버 summary 반영이 늦을 수 있어 history 로 확인
        try:
            h = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}").history(keys=["train_step"], samples=100000, pandas=False)
            done = max((r["train_step"] for r in h), default=None)
        except Exception:
            done = None
    done = -1 if done is None else int(done)
    print(f"[wandb_sync] {run.url}  (이어쓰기: step {done} 까지 업로드됨)", flush=True)
    up = Uploader(run, per_step)
    src = RunDirSource(a.source, up, done, decode=not a.no_decode) if is_dir else LogSource(a.source, up, done, 2, a.sample_every)
    while True:
        stop = a.until_pid is not None and not _alive(a.until_pid)
        src.poll(final=not a.follow or stop)
        up.flush_summary()
        if not a.follow or stop:
            break
        time.sleep(a.interval)
    print(f"[wandb_sync] 완료: step {src.done}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
