# Stage 1 — oracle on demand: gate = FAIL

Base Tar-7B (no LoRA). 300 prompts (held-out 100 + 200; 298 unique qids — 2 CoVA duplicate questions).
main = leaky excluded (33 leaky by text heuristic) → 265. Full tables: `report.md`. Raw: `logs/stage1/base/records.jsonl`.
Scripts: `scripts/stage1_prep.py`, `scripts/oracle_eval.py`, `scripts/stage1_report.py`.

Conditions: text_only (neutral prompt, `<im_start>` banned) / model_drawn (sketch prompt, own image + bridge) /
oracle (sketch prompt; at `<im_start>` the model's image is replaced by the CoVA first image, S2 = 81 tokens, + bridge) /
corrupt (same, another prompt's GT image). The three image conditions share the prefix up to `<im_start>`.

## Gate (free-form, prompts where `<im_start>` was emitted, non-leaky)

| | greedy (n=89) | pass@1 (n=106) | pass@8 (n=106) |
|---|---|---|---|
| text_only | 25.8 | 22.5 | 64.2 |
| oracle | 25.8 | 19.6 | 56.6 |
| corrupt | 28.1 | 22.3 | 61.3 |
| oracle − text_only | +0.0 (McNemar p=1) | −2.9 [−6.2, +0.2] | −7.5 [−17.0, +1.9] |
| oracle − corrupt | −2.2 (p=0.80) | −2.7 [−5.7, +0.1] | −4.7 [−13.2, +2.8] |

Sample-level (emitting samples only, exact pairing via shared prefix, free-form): oracle 20.6 vs corrupt 24.2 (−3.6 [−7.5, 0.0]).
Oracle does not beat either baseline; point estimates are negative.

## Other observations

- `<im_start>` position: 180/184 greedy and 1175/1200 sampled emissions are at token 0 — the model draws before any text.
  "On demand" is in practice "an image before reasoning".
- Emission: greedy 184/298 prompts, sampled 1200/2384 samples.
- Oracle > model_drawn (+4–7 pp) but this is largely an artifact: at T=0.8, 444/1216 model images used S1/S0 scale
  (169/729 tokens) and hit the 96-token image cap → no answer (model_drawn answer rate 76% vs ~94% elsewhere). Greedy images are all S2.
- MC: the sketch prompt + any image is worse than text_only (greedy corrupt − text_only −14 pp, p=0.002).
- Not-emitted prompts: sketch prompt alone is ~5 pp below the neutral prompt (greedy 16.2 vs 21.2; pass@1 18.0 vs 22.8).

## Caveats

- S2 (81 tokens) round-trip keeps layout but destroys labels (`../stage1_roundtrip.png`), so the oracle carries structure only.
  An S0 (729-token) oracle would exceed the 768-token budget and was not run.
- Leaky check is text-only (gold value in the trace before / right after the first `<img>`); images were not inspected for written answers.
- text_only uses the neutral prompt (as specified), the others the sketch prompt — the prompt difference is part of the text_only gap.
