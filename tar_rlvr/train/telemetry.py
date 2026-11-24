"""학습 텔레메트리: "학습이 실제로 일어나고 있나" 를 보는 지표를 logs/runs/<run>/ 에 JSONL 로 남긴다.

학습 프로세스는 파일만 쓴다 (네트워크·wandb 의존 없음 → 업로드 문제로 학습이 죽지 않음).
W&B 업로드·이동평균·이미지 디코딩은 scripts/wandb_sync.py 사이드카가 한다.

  metrics.jsonl        {"step": s, "<section>/<name>": value, ...}  — 한 줄에 여러 지표, 같은 step 이 여러 줄일 수 있음
  samples/step_XXXXXX.jsonl   completion 원문 (시각 토큰 포함) — sample_every step 마다 ~10개
  eval/step_XXXXXX.jsonl      held-out eval 의 completion 원문

섹션:
  learning_signal/  학습 가능한 그룹 비율, |advantage|, entropy(텍스트/시각), KL(ref), clip 비율, grad norm, pass@k
  visual/           그룹 내 그림 유/무 정답률 차, 그림 샘플의 advantage, <im_start> 확률(정책 성향), 그림 수·위치·토큰 수
  length/           정답/오답별 길이, 시각 토큰 뺀 길이, 잘림 비율
  verifier/         reward 0 의 원인 분해 (파싱 실패 / 형식 / 오답), 규칙 기반 vs math-verify 불일치
  eval/             held-out greedy / pass@k

Tunix 연결 지점 (tunix 0.1.x):
  - metric_fn        : 롤아웃·advantage 직후 (completion 텍스트, reward, advantage)
  - generate 래퍼    : completion 토큰 → 길이·잘림·그림 위치 (metric_fn 은 텍스트만 받으므로 직전 호출 결과를 넘겨받음)
  - loss 래퍼        : 학습 forward 의 logits 로 entropy(텍스트/시각) 와 P(<im_start>) — 추가 forward 없음
  - advantage 래퍼   : beta=0 이어도 ref logps 를 계산해 KL 을 기록 (loss 에는 안 들어감), 그리고 eval 트리거
  - 로거 백엔드      : Tunix 가 jax.monitoring 으로 내보내는 스칼라(loss, grad_norm, clip, kl …)를 받아 이름을 섹션으로 옮김
"""
from __future__ import annotations
import collections, dataclasses, json, os, random, threading, time, warnings
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")   # 그림 샘플이 없는 step 의 nanmean

from tar_rlvr.rewards import _refs, _s
from tar_rlvr.rewards.parsing import ANSWER_RE, boxed_contents, extract_answer
from tar_rlvr.rewards.verifiers import VERIFIERS, verifier_breakdown, _canon, _norm

NAN = float("nan")


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size and not np.all(np.isnan(x)) else NAN


def _median(x):
    return float(np.median(x)) if len(x) else NAN


# ---------------------------------------------------------------------------------------------- 토큰 정보
@dataclasses.dataclass
class TokInfo:
    pad: int
    eos: int
    vis_lo: int            # 시각 계열 토큰 범위 [vis_lo, vis_hi): <im_start> <im_end> <S0..2> <I0..>
    vis_hi: int
    img_lo: int            # <I0> — 실제 그림 코드 토큰 시작
    im_start: int | None   # tiny 토크나이저엔 없음


def token_info(tok, pad: int, eos: int, num_visual_tokens: int) -> TokInfo:
    if hasattr(tok, "convert_tokens_to_ids"):
        ims, i0 = tok.convert_tokens_to_ids("<im_start>"), tok.convert_tokens_to_ids("<I0>")
        return TokInfo(pad, eos, ims, i0 + num_visual_tokens, i0, ims)
    v0 = tok.vis0          # tiny ByteTokenizer
    return TokInfo(pad, eos, v0, v0 + num_visual_tokens, v0, None)


def token_stats(rows, ti: TokInfo, budget: int) -> list[dict]:
    """completion 토큰 행마다: 길이(EOS 제외), 잘림(EOS 없이 끝), 시각/텍스트 토큰 수, 그림 수, 첫 그림 상대 위치."""
    out = []
    for r in rows:
        r = np.asarray(r).tolist()
        n, ended = 0, False
        for t in r:
            if t == ti.eos:
                ended = True
                break
            if t == ti.pad:
                break
            n += 1
        seq = np.asarray(r[:n], dtype=np.int64)
        vis = (seq >= ti.vis_lo) & (seq < ti.vis_hi)
        starts = np.flatnonzero(seq == ti.im_start) if ti.im_start is not None else np.array([], int)
        out.append(dict(n=n, truncated=(not ended) and n >= budget - 1, n_vis=int(((seq >= ti.img_lo) & (seq < ti.vis_hi)).sum()),
                        n_text=int(n - vis.sum()), n_img=int(len(starts)),
                        first_img=(float(starts[0]) / max(1, n)) if len(starts) else NAN))
    return out


# ---------------------------------------------------------------------------------------------- 싱크/백엔드
class Sink:
    """logs/runs/<run>/metrics.jsonl 에 한 줄씩. reward fn 이 스레드에서 돌 수 있어 lock."""

    def __init__(self, run_dir: str):
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._f = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, step: int, metrics: dict):
        rec = {"step": int(step), "t": round(time.time(), 1)}
        for k, v in metrics.items():
            try:
                v = float(np.asarray(v))
            except Exception:
                continue
            rec[k] = None if np.isnan(v) or np.isinf(v) else v
        with self._lock:
            self._f.write(json.dumps(rec) + "\n")
            self._f.flush()

    def write_rows(self, sub: str, step: int, rows: list[dict]):
        d = os.path.join(self.dir, sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"step_{step:06d}.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Tunix 스칼라 이름 → 섹션 이름. event = "<prefix>/<mode>/<name>"
_ACTOR = {"loss": "learning_signal/loss", "grad_norm": "learning_signal/grad_norm",
          "pg_clipfrac": "learning_signal/clip_frac", "kl": "learning_signal/kl_ref",
          "entropy": "learning_signal/entropy", "entropy_text": "learning_signal/entropy_text",
          "entropy_visual": "learning_signal/entropy_visual", "p_im_start": "visual/p_im_start",
          "learning_rate": "learning_signal/lr"}
_OTHER = {"rewards/score/mean": "learning_signal/reward_mean", "rewards/score/max": "learning_signal/reward_max",
          "rewards/score/min": "learning_signal/reward_min"}
_PREFIX_MAP = {"rewards": "learning_signal/reward_", "completions": "length/"}   # 그 밖의 Tunix 기본 지표
_DROP_PREFIX = ("debug", "global")


class TunixBackend:
    """metrax LoggingBackend 프로토콜 (log_scalar, close). Tunix 의 MetricsLogger 가 jax.monitoring 리스너로 등록."""

    def __init__(self, sink: Sink):
        self.sink = sink

    def log_scalar(self, event: str, value, **kw):
        parts = event.split("/")
        if len(parts) < 3 or parts[1] not in ("train", "Mode.TRAIN"):
            return
        prefix, name = parts[0], "/".join(parts[2:])
        if prefix in _DROP_PREFIX:
            return
        step = kw.get("step")
        step = -1 if step is None else int(step)
        if prefix == "actor":
            key = _ACTOR.get(name)
            if key is None:
                return
            step -= 1      # 학습 step k(1부터) 는 롤아웃 step k-1 의 데이터로 업데이트 → 롤아웃 지표와 같은 step 에 둔다
        elif f"{prefix}/{name}" in _OTHER:
            key = _OTHER[f"{prefix}/{name}"]
        elif prefix in _PREFIX_MAP:
            key = _PREFIX_MAP[prefix] + name.replace("/", "_")
        else:
            key = f"{prefix}/{name}"
        self.sink.write(step, {key: value})

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------- 본체
class Telemetry:
    def __init__(self, cfg: dict, tok, run_dir: str):
        t = cfg.get("telemetry", {})
        g = cfg["grpo"]
        self.cfg, self.tok, self.G = cfg, tok, g["num_generations"]
        self.budget = g["max_completion_length"]
        self.sample_every = t.get("sample_every", 50)
        self.n_samples = t.get("n_samples", 10)
        self.token_metrics = t.get("token_metrics", True)
        self.log_kl = t.get("log_kl", True)
        ev = cfg.get("eval", {})
        self.eval_every, self.eval_k = ev.get("every", 25), ev.get("pass_k", 8)
        self.eval_k_prompts = ev.get("k_prompts")          # None = 전체
        self.eval_at_start = ev.get("at_start", True)
        self.eval_batch = ev.get("batch", 16)
        self.sink = Sink(run_dir)
        self.backend = TunixBackend(self.sink)
        self.cluster = self.ti = None
        self.eval_rows: list[dict] = []
        self._last_gen: list[dict] | None = None
        self._eval_done: set[int] = set()
        self._rng = random.Random(0)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1, default=str)

    def metrics_logging_options(self):
        from tunix.sft import metrics_logger
        backend = self.backend
        return metrics_logger.MetricsLoggerOptions(log_dir=self.sink.dir, backend_kwargs={"custom_backend": [lambda: backend]})

    @property
    def step(self) -> int:
        return int(self.cluster.global_steps) if self.cluster is not None else 0

    # ------------------------------------------------------------------ 설치
    def attach(self, cluster, learner, eval_rows: list[dict] | None = None):
        self.cluster = cluster
        self.eval_rows = eval_rows or []
        self.ti = token_info(self.tok, cluster.rollout.pad_id(), cluster.rollout.eos_id(),
                             self.cfg["model"]["num_visual_tokens"])
        self._wrap_generate(cluster)
        if self.token_metrics:
            self._wrap_loss(cluster)
        self._wrap_advantage(cluster, learner)

    def _wrap_generate(self, cluster):
        from tunix.rl import rl_cluster as rl_cluster_lib
        inner = cluster.generate

        def generate(prompts, *a, mode=rl_cluster_lib.Mode.TRAIN, **kw):
            out = inner(prompts, *a, mode=mode, **kw)
            if mode == rl_cluster_lib.Mode.TRAIN:
                self._last_gen = token_stats(out.tokens, self.ti, self.budget)
            return out

        cluster.generate = generate

    def _wrap_loss(self, cluster):
        """GRPO loss 가 이미 만드는 logits 를 가로채 entropy(텍스트/시각)·P(<im_start>) 를 aux 에 추가.
        logits 는 rollout 온도로 나눈 값이라 P(<im_start>) 는 실제 샘플링 분포 기준이다."""
        import jax, jax.numpy as jnp
        from tunix.rl import algo_core, common
        trainer = cluster.actor_trainer
        base_loss = trainer.loss_fn
        ti = self.ti

        def loss_fn(model, train_example, algo_config):
            cap = {}
            orig = common.compute_per_token_logps

            def spy(*a, **k):
                out = orig(*a, **k)
                if k.get("return_logits"):
                    cap["logits"] = out[1]
                return out
            algo_core.common.compute_per_token_logps = spy
            try:
                loss, aux = base_loss(model, train_example, algo_config)
            finally:
                algo_core.common.compute_per_token_logps = orig
            aux = dict(aux)
            if "logits" in cap:
                logits = jax.lax.stop_gradient(cap["logits"]).astype(jnp.float32)
                ids, mask = train_example.completion_ids, train_example.completion_mask.astype(bool)
                vis = (ids >= ti.vis_lo) & (ids < ti.vis_hi)
                ent = algo_core.compute_entropy_from_logits(logits)

                def mmean(x, m):
                    m = m.astype(jnp.float32)
                    return jnp.where(m.sum() > 0, (x * m).sum() / jnp.maximum(m.sum(), 1.0), jnp.nan)
                aux["entropy_text"] = mmean(ent, mask & ~vis)
                aux["entropy_visual"] = mmean(ent, mask & vis)
                if ti.im_start is not None:
                    lse = jax.nn.logsumexp(logits, axis=-1)
                    aux["p_im_start"] = mmean(jnp.exp(logits[..., ti.im_start] - lse), mask & ~vis)
                else:
                    aux["p_im_start"] = jnp.float32(jnp.nan)
            return loss, aux

        trainer.with_loss_fn(loss_fn, has_aux=True)
        trainer.with_rl_metrics_to_log({"kl": np.mean, "pg_clipfrac": np.mean, "entropy": np.mean,
                                        "entropy_text": np.nanmean, "entropy_visual": np.nanmean,
                                        "p_im_start": np.nanmean})

    def _wrap_advantage(self, cluster, learner):
        orig = learner._generate_and_compute_advantage
        from tunix.rl import rl_cluster as rl_cluster_lib
        G = self.G

        def gen_adv(training_input, mode=rl_cluster_lib.Mode.TRAIN):
            if mode == rl_cluster_lib.Mode.TRAIN:
                self._maybe_eval()
            ex = orig(training_input, mode)
            if self.log_kl and ex.ref_per_token_logps is None and self._has_reference():
                # beta=0 이면 Tunix 는 ref logps 를 안 구한다 → 기록용으로만 계산 (loss 에는 beta*kl = 0)
                ref = cluster.get_ref_per_token_logps(
                    prompt_tokens=ex.prompt_ids, completion_tokens=ex.completion_ids,
                    pad_id=cluster.rollout.pad_id(), eos_id=cluster.rollout.eos_id(),
                    micro_batch_size=(self.cfg["grpo"].get("logps_micro_batch_size") or 1) * G)
                ex = ex.replace(ref_per_token_logps=ref)
            return ex

        learner._generate_and_compute_advantage = gen_adv

    def _has_reference(self) -> bool:
        c = self.cluster
        if getattr(c, "reference", None) is not None:
            return True
        try:                                   # Tunix 는 초기화 후 reference 를 inference_worker 로 옮긴다
            return c.inference_worker.get_model("reference") is not None
        except Exception:
            return False

    # ------------------------------------------------------------------ 롤아웃 지표 (Tunix metric_fn)
    def metric_fn(self, prompts, completions, rewards, advantages=None, advances=None, **cols):
        adv = np.asarray(advantages if advantages is not None else advances, dtype=float)
        comps = [_s(c) for c in completions]
        r = np.asarray(rewards, dtype=float)
        ok = r >= 1.0
        refs, tasks = _refs(cols), [_s(t) for t in cols["task"]]
        n, G = len(comps), self.G
        ts = self._last_gen if self._last_gen is not None and len(self._last_gen) == n else None
        self._last_gen = None
        has_img = np.array(["<im_start>" in c for c in comps])
        if ts is not None and self.ti.im_start is not None:
            has_img = np.array([t["n_img"] > 0 for t in ts])
        ans = [extract_answer(c) for c in comps]
        tagged = [bool(ANSWER_RE.search(c) or boxed_contents(c)) for c in comps]

        # 그룹 단위
        learnable, all0, all1, pass_k, maj, paired = [], [], [], [], [], []
        for s in range(0, n, G):
            rr, oo, hh = r[s:s + G], ok[s:s + G], has_img[s:s + G]
            learnable.append(float(rr.std() > 1e-8)); all0.append(float(not oo.any())); all1.append(float(oo.all()))
            pass_k.append(float(oo.any()))
            votes = collections.Counter(_canon(_norm(a)) for a in ans[s:s + G] if a)
            if votes:
                top = votes.most_common(1)[0][0]
                maj.append(float(any(oo[j] for j in range(G) if ans[s + j] and _canon(_norm(ans[s + j])) == top)))
            else:
                maj.append(0.0)
            paired.append(float(oo[hh].mean() - oo[~hh].mean()) if hh.any() and (~hh).any() else NAN)

        # verifier: reward 0 분해 + 규칙/math-verify 불일치
        parse_fail = [(not ok[i]) and ans[i] is None for i in range(n)]
        fmt_fail = [(not ok[i]) and ans[i] is not None and not tagged[i] for i in range(n)]
        wrong = [(not ok[i]) and ans[i] is not None and tagged[i] for i in range(n)]
        dis, rule_only, mv_only = [], [], []
        for i in range(n):
            if ans[i] is None or VERIFIERS.get(tasks[i]) is None:
                continue
            rule, mv = verifier_breakdown(ans[i], refs[i])
            if mv is None:
                continue
            dis.append(float(rule != mv)); rule_only.append(float(rule > mv)); mv_only.append(float(mv > rule))

        m = {
            "learning_signal/acc": (float(ok.mean()), np.mean),
            "learning_signal/learnable_frac": (float(np.mean(learnable)), np.mean),
            "learning_signal/all_zero_frac": (float(np.mean(all0)), np.mean),
            "learning_signal/all_one_frac": (float(np.mean(all1)), np.mean),
            "learning_signal/adv_abs_mean": (float(np.abs(adv).mean()), np.mean),
            "learning_signal/pass_at_1": (float(ok.mean()), np.mean),
            f"learning_signal/pass_at_{G}": (float(np.mean(pass_k)), np.mean),
            "learning_signal/maj_vote_acc": (float(np.mean(maj)), np.mean),
            "visual/acc_vis": (float(ok[has_img].mean()) if has_img.any() else NAN, np.nanmean),
            "visual/acc_txt": (float(ok[~has_img].mean()) if (~has_img).any() else NAN, np.nanmean),
            "visual/paired_acc_diff": (_nanmean(paired), np.nanmean),
            "visual/paired_group_frac": (float(np.mean([not np.isnan(p) for p in paired])), np.mean),
            "visual/adv_img_mean": (float(adv[has_img].mean()) if has_img.any() else NAN, np.nanmean),
            "visual/img_frac": (float(has_img.mean()), np.mean),
            "length/chars_median": (_median([len(c) for c in comps]), np.mean),
            "length/boxed_only_frac": (float(np.mean([bool(_BOXED_ONLY(c)) for c in comps])), np.mean),
            "verifier/zero_parse_fail": (float(np.mean(parse_fail)), np.mean),
            "verifier/zero_format_fail": (float(np.mean(fmt_fail)), np.mean),
            "verifier/zero_wrong_answer": (float(np.mean(wrong)), np.mean),
            "verifier/rule_mv_disagree": (_nanmean(dis) if dis else NAN, np.nanmean),
            "verifier/rule_only_correct": (_nanmean(rule_only) if rule_only else NAN, np.nanmean),
            "verifier/mv_only_correct": (_nanmean(mv_only) if mv_only else NAN, np.nanmean),
        }
        if ts is not None:
            L = np.array([t["n"] for t in ts]); Lt = np.array([t["n_text"] for t in ts])
            m.update({
                "visual/images_per_completion": (float(np.mean([t["n_img"] for t in ts])), np.mean),
                "visual/visual_tokens_per_completion": (float(np.mean([t["n_vis"] for t in ts])), np.mean),
                "visual/first_img_relpos": (_nanmean([t["first_img"] for t in ts]), np.nanmean),
                "length/tokens_median": (_median(L), np.mean),
                "length/text_tokens_median": (_median(Lt), np.mean),
                "length/tokens_median_correct": (_median(L[ok]) if ok.any() else NAN, np.nanmean),
                "length/tokens_median_incorrect": (_median(L[~ok]) if (~ok).any() else NAN, np.nanmean),
                "length/truncation_rate": (float(np.mean([t["truncated"] for t in ts])), np.mean),
            })

        step = self.step
        if self.sample_every and step % self.sample_every == 0:
            per_call = max(1, self.n_samples // max(1, self.cfg["grpo"]["batch_size"]))
            idx = self._rng.sample(range(n), min(per_call, n))
            img_idx = [i for i in range(n) if has_img[i] and i not in idx]
            if img_idx and not any(has_img[i] for i in idx):   # 그림 샘플이 하나는 들어가게
                idx[-1] = img_idx[0]
            self.sink.write_rows("samples", step, [dict(
                step=step, prompt=_s(prompts[i])[-1500:], completion=comps[i], reward=float(r[i]), adv=float(adv[i]),
                has_image=bool(has_img[i]), length=(ts[i]["n"] if ts else len(comps[i])), extracted=ans[i],
                ref=refs[i]) for i in idx])
        return m

    # ------------------------------------------------------------------ held-out eval
    def _maybe_eval(self):
        step = self.step
        if not self.eval_rows or not self.eval_every or step in self._eval_done:
            return
        if not ((step == 0 and self.eval_at_start) or (step > 0 and step % self.eval_every == 0)):
            return
        self._eval_done.add(step)
        t0 = time.time()
        try:
            res = self.run_eval(step)
        except Exception as e:                       # eval 실패로 학습을 죽이지 않는다
            print(f"[telemetry] eval step {step} 실패: {type(e).__name__}: {e}", flush=True)
            return
        res["eval/seconds"] = time.time() - t0
        self.sink.write(step, res)
        print(f"[telemetry] eval step {step}: " + ", ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in res.items()), flush=True)

    def _generate(self, prompts: list[str], **cfg_over) -> list[tuple[str, dict]]:
        out = []
        gen = getattr(self.cluster, "generate_with_config", None) or self._plain_generate
        for s in range(0, len(prompts), self.eval_batch):
            o = gen(prompts[s:s + self.eval_batch], **cfg_over)
            st = token_stats(o.tokens, self.ti, self.budget)
            out += list(zip([_s(t) for t in o.text], st))
        return out

    def _plain_generate(self, prompts, **overrides):
        """continuation 이 꺼진 경우의 eval 생성."""
        from tunix.rl import rl_cluster as rl_cluster_lib
        cfg = dataclasses.replace(self.cluster.cluster_config.rollout_config, **overrides)
        with self.cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ROLLOUT]:
            return self.cluster.rollout.generate(list(prompts), cfg)

    def run_eval(self, step: int) -> dict:
        rows, G = self.eval_rows, self.eval_k
        prompts = [r["prompts"] for r in rows]

        def score(c, row):
            a = extract_answer(c)
            return VERIFIERS[_s(row["task"])](a, json.loads(_s(row["ref"]))) if a is not None else 0.0
        greedy = self._generate(prompts, top_k=1, temperature=1.0)
        g_ok = np.array([score(c, row) >= 1 for (c, _), row in zip(greedy, rows)])
        g_img = np.array([st["n_img"] > 0 for _, st in greedy])
        res = {"eval/greedy_acc": g_ok.mean(), "eval/greedy_img_frac": g_img.mean(),
               "eval/greedy_tokens_median": _median([st["n"] for _, st in greedy]),
               "eval/greedy_truncation_rate": np.mean([st["truncated"] for _, st in greedy]),
               "eval/n_prompts": len(rows)}
        recs = [dict(kind="greedy", qid=row.get("qid"), completion=c, correct=bool(o), n_tokens=st["n"])
                for (c, st), row, o in zip(greedy, rows, g_ok)]
        if G and G > 1:
            krows = rows[: self.eval_k_prompts] if self.eval_k_prompts else rows
            sampled = self._generate([r["prompts"] for r in krows for _ in range(G)])
            ok = np.array([score(c, row) >= 1 for (c, _), row in zip(sampled, [r for r in krows for _ in range(G)])])
            ok = ok.reshape(len(krows), G)
            img = np.array([st["n_img"] > 0 for _, st in sampled]).reshape(len(krows), G)
            res.update({"eval/pass_at_1": ok.mean(), f"eval/pass_at_{G}": ok.any(1).mean(),
                        "eval/sampled_img_frac": img.mean(), "eval/n_prompts_k": len(krows)})
            recs += [dict(kind="sampled", qid=krows[i // G].get("qid"), completion=c, correct=bool(ok.flat[i]),
                          n_tokens=st["n"]) for i, (c, st) in enumerate(sampled)]
        self.sink.write_rows("eval", step, recs)
        return res


import re as _re
_BOXED_ONLY_RE = _re.compile(r"^\s*\$*\s*\\boxed\{.*\}\s*\$*\s*\.?\s*$", _re.S)


def _BOXED_ONLY(c: str) -> bool:
    return bool(_BOXED_ONLY_RE.match(c))
