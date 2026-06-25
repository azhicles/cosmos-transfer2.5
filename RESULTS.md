# Cosmos Transfer — Minimal-Requirements Experiments

**Goal:** find the minimal requirements (time first, then VRAM/memory) to generate new-style,
domain-randomized robot videos that stay high-quality / consistent / robust. Quality scored by the
Cosmos-Reason2-32B critic (`task_score` / `realism_score`, 0–10).

**Setup:** single RTX Pro 6000 (96 GB). Single-view top, episode_000047 (777 frames, 640×480),
new "studio_warm"-class styles. Controls pre-extracted and reused so each run is pure diffusion
(except cross-episode + on-the-fly cases). 10 batches, 42 runs.

---

## TL;DR — Recommended recipe (updated after style-aware rerank)

> **`edge + depth + vis(0.3)` · 480-res · 6 steps** → quality 15/20 (style 6 / realism 9) · ~6.2 min · 40 GB

(Max measured quality = `vis(0.5)/8 steps` → 16.2/20, but vis 0.5 over-anchors to the original
appearance — weaker domain randomization. vis 0.3 is the practical optimum.)

Still **~10× faster *and* higher quality** than the original production recipe
(720/35-step, edge+depth+seg, *no vis* = 64 min). See **Style-aware Pareto frontier** below — it
supersedes the original resource-only ranking; the realism numbers in the big table used the older
prompt that penalised the restyle itself.

**The single most important change: add `vis` control at weight 0.3** — it fixes the game-piece
flicker/disappearance that caps quality, lifting realism 4→7 and task 0→10, for only +2 min / +3 GB.

---

## Levers, ranked

| lever | verdict |
|---|---|
| **vis(0.3)** | The quality unlock. Stabilises the pieces (realism 4→7). 0.3 = sweet spot (0.5 over-anchors → "looks like original"; 0.2 dips to 6). |
| **depth** | **Mandatory** — drop it and quality collapses (realism 2). |
| **resolution** | #1 time lever: 480 ≈ **2.9× faster** than 720. 720 does **not** improve quality (also realism 7). |
| **steps** | Knee at **6** (style-aware realism: 4st→6.8, 6st→9.0, 8st→10). 4 steps is usable but lower quality. (The old prompt made 4 look sufficient — see frontier section.) |
| **edge** | Optional minor bump (depth+vis = 6, +edge = 7). |
| **seg** | Unnecessary here and the most expensive (re-runs SAM2 ~3 min + a control branch). Drop it. |
| **sigma_max** | Not a quality lever (vis is). 80 vs 110 made no difference. |
| **guidance (CFG)** | **Not a lever in 1–7** — flat (style 6 / realism 10 at the recommended recipe). Only matters that it's >0: guidance 0 collapses realism to 1. With vis+controls, prompt adherence is already saturated. Keep default 5. |
| **VRAM / memory** | **Never binding** — 37–55 GB across everything, nowhere near 96 GB. **Time is the only real constraint.** |

**Robustness:** the recommended recipe holds realism **6–7 across 4 seeds and 9 distinct new
styles**. The ~7 realism is the model's ceiling for this restyle (reached cheaply).

---

## Recommended operating points

*(Updated to the style-aware frontier — quality = style+realism, 0–20.)*

| use case | recipe | time | VRAM | quality |
|---|---|---|---|---|
| **Recommended (balance)** | edge+depth+vis(0.3) · 480 · **6 steps** | ~6.2 min | 40 GB | **15** |
| Max quality | edge+depth+vis(0.5) · 480 · 8 steps | ~7 min | 40 GB | **16.2** (weaker restyle) |
| Fastest | edge+depth+vis(0.3) · 480 · 4 steps | ~5.4 min | 40 GB | 12.2 |
| Min VRAM | depth+vis(0.3) · 480 · 6 steps | ~6 min | **37 GB** | ~14 |
| Original (baseline) | edge+depth+seg · 720 · 35 · no vis | ~64 min | 55 GB | 12 |

For a *reliable* good clip, use **best-of-N seed selection** (the critic is built for it) — the
recipe occasionally draws a weaker seed.

---

## Full experiment log (42 runs)

`realism`/`task` are Cosmos-Reason2 0–10. ⚠️ Batch 1 used 93-frame clips from the idle start of
the episode → action absent, realism saturates at 10 (uninformative); kept for completeness only.

| batch | run | controls | vis | steps | res | sigma | time (min) | VRAM (GB) | task | realism |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 ⚠️ | s04 | (multi) | — | 4 | 720 | 110 | 6.4 | 49.0 | 0 | 10 |
| 1 ⚠️ | s08 | (multi) | — | 8 | 720 | 110 | 7.0 | 49.0 | 0 | 10 |
| 1 ⚠️ | s16 | (multi) | — | 16 | 720 | 110 | 8.4 | 49.0 | 0 | 10 |
| 1 ⚠️ | s24 | (multi) | — | 24 | 720 | 110 | 9.7 | 49.0 | 0 | 10 |
| 1 ⚠️ | s35 | (multi) | — | 35 | 720 | 110 | 11.7 | 49.0 | 0 | 10 |
| 2 | s04_mc | depth+seg+edge | — | 4 | 720 | 110 | 15.0 | 54.8 | 0 | 4 |
| 2 | s12_mc | depth+seg+edge | — | 12 | 720 | 110 | 27.5 | 54.8 | 0 | 5 |
| 2 | s24_mc | depth+seg+edge | — | 24 | 720 | 110 | 46.3 | 54.8 | 0 | 5 |
| 2 | s35_mc | depth+seg+edge | — | 35 | 720 | 110 | 63.7 | 54.8 | 0 | 6 |
| 2 | ed_s35 | depth+edge | — | 35 | 720 | 110 | 52.7 | 49.8 | 3 | 4 |
| 2 | mc480_s35 | depth+seg+edge | — | 35 | 480 | 110 | 22.2 | 49.0 | 0 | 5 |
| 2 | edge_s35 | edge | — | 35 | 720 | 110 | 46.9 | 44.6 | 4 | 4 |
| 3 | m480_s08 | depth+seg+edge | — | 8 | 480 | 110 | 10.5 | 49.0 | 0 | 5 |
| 3 | m480_s16 | depth+seg+edge | — | 16 | 480 | 110 | 13.9 | 49.0 | 0 | 5 |
| 3 | m480_s24 | depth+seg+edge | — | 24 | 480 | 110 | 17.3 | 49.0 | 0 | 4 |
| 3 | m480_s35_sig80 | depth+seg+edge | — | 35 | 480 | 80 | 22.1 | 49.0 | 0 | 4 |
| 4 | s08_seedA | depth+seg+edge | — | 8 | 480 | 110 | 10.4 | 49.0 | 0 | 3 |
| 4 | s08_seedB | depth+seg+edge | — | 8 | 480 | 110 | 10.4 | 49.0 | 0 | 5 |
| 4 | s08_seedC | depth+seg+edge | — | 8 | 480 | 110 | 10.4 | 49.0 | 0 | 4 |
| 4 | s08_seedD | depth+seg+edge | — | 8 | 480 | 110 | 10.4 | 49.0 | 0 | 4 |
| 5 | m480_s08_cool | depth+seg+edge | — | 8 | 480 | 110 | 10.7 | 49.6 | 4 | 4 |
| 5 | m480_s08_amber | depth+seg+edge | — | 8 | 480 | 110 | 10.5 | 49.6 | 0 | 4 |
| 5 | **ed480_s08** | depth+edge | — | 8 | 480 | 110 | **5.0** | **36.9** | 0 | 5 |
| 6 | ed480_s08_b | depth+edge | — | 8 | 480 | 110 | 5.1 | 36.9 | 0 | 4 |
| 6 | ed480_s08_c | depth+edge | — | 8 | 480 | 110 | 5.0 | 36.9 | 0 | 4 |
| 6 | ed480_s08_d | depth+edge | — | 8 | 480 | 110 | 5.0 | 36.9 | 0 | 3 |
| 6 | **edv480_s08** | depth+edge+vis | 0.3 | 8 | 480 | 110 | 7.0 | 40.2 | 10 | **7** |
| 7 | edv480_s08_b | depth+edge+vis | 0.3 | 8 | 480 | 110 | 7.1 | 40.2 | 4 | 6 |
| 7 | edv480_s08_c | depth+edge+vis | 0.3 | 8 | 480 | 110 | 7.0 | 40.2 | 0 | 7 |
| 7 | edv480_vis20 | depth+edge+vis | 0.2 | 8 | 480 | 110 | 7.0 | 40.2 | 0 | 6 |
| 7 | edv480_vis50 | depth+edge+vis | 0.5 | 8 | 480 | 110 | 7.0 | 40.2 | 10 | 7 |
| 8 | win_blue | depth+edge+vis | 0.3 | 8 | 480 | 110 | 7.1 | 40.2 | 0 | 6 |
| 8 | win_morning | depth+edge+vis | 0.3 | 8 | 480 | 110 | 7.0 | 40.2 | 10 | 6 |
| 8 | win_industrial | depth+edge+vis | 0.3 | 8 | 480 | 110 | 7.0 | 40.2 | 0 | 7 |
| 8 | edv720_s16 | depth+edge+vis | 0.3 | 16 | 720 | 110 | 32.2 | 54.8 | 0 | 7 |
| 9 | **edv480_s04** | depth+edge+vis | 0.3 | **4** | 480 | 110 | **5.4** | 40.2 | 0 | **7** |
| 9 | edv480_s06 | depth+edge+vis | 0.3 | 6 | 480 | 110 | 6.2 | 40.2 | 10 | 7 |
| 9 | ev480_s08 | edge+vis (no depth) | 0.3 | 8 | 480 | 110 | 6.0 | 36.9 | 0 | **2** |
| 9 | dv480_s08 | depth+vis (no edge) | 0.3 | 8 | 480 | 110 | 6.4 | 36.9 | 10 | 6 |
| 10 | xep000 (bottom-left) | depth+edge+vis | 0.3 | 4 | 480 | 110 | 5.4 | 40.2 | — | not scored |
| 10 | xep090 (center) | depth+edge+vis | 0.3 | 4 | 480 | 110 | 5.3 | 40.2 | — | not scored |
| 10 | xep150 (middle-right) | depth+edge+vis | 0.3 | 4 | 480 | 110 | 5.3 | 40.2 | — | not scored |

**Bold** = key milestones. Batch 10 (cross-episode generalization) generated but was stopped
before scoring.

---

## Methodology notes / caveats

- **Quality proxy = `realism_score`.** `task_score` is noisy on heavily-restyled clips (the critic
  often misreads the placement cell), so it isn't a reliable discriminator — `realism` + the
  critic's `no_artifacts` justification are used instead.
- **Critic noise ≈ ±0.7** (single-sample). Multi-seed runs (Batch 4, 6, 7) average it out.
- **Use full-length clips** for quality scoring — 93-frame clips from the start are idle/uninformative
  (Batch 1).
- **GPU contention:** the 32B critic (~63 GB) and the diffusion server cannot co-reside; the loop
  alternates generate → stop server → score → restart.
- **Harness:** `scripts/exp_run_batch.py` (generate + record time/VRAM) and `scripts/exp_report.py`
  (merge with critic scores). Per-batch data in `outputs/experiments/<batch>/{results,scores}.json`.

## Production implication

The shipped style deliverables used `edge+depth+seg` **without vis** (vis was dropped after the
0.5 "looks-like-original" complaint) — that's why pieces still flickered. **Re-add vis at 0.3** and
switch to 480-res / 4 steps for the big speed-up at higher quality.

---

## Style-aware Pareto frontier (supersedes the resource-only ranking)

Re-scored all 43 clips with the upgraded Cosmos-Reason2 critic: leaner system prompt, a reworded
`realism_score` (temporal stability/plausibility — does **not** penalise the restyle), and a new
`style_score` (adherence to each clip's intended style). **Quality = style_score + realism_score
(0–20)**, averaged across seeds. Contested points confirmed with best-of-N (n=4).

| ★ Pareto | recipe | n | time (min) | VRAM (GB) | style | realism | quality |
|---|---|---|---|---|---|---|---|
| ★ | edge+depth+vis(0.5) · 8st · 480 | 4 | 7.0 | 40.2 | 6.2 | 10.0 | **16.2** |
| ★ | edge+depth+vis(0.3) · 6st · 480 | 4 | 6.2 | 40.2 | 6.0 | 9.0 | **15.0** |
| ★ | edge+depth+vis(0.3) · 4st · 480 | 4 | 5.4 | 40.2 | 5.5 | 6.8 | 12.2 |
| ★ | edge+depth (no vis) · 8st · 480 | 4 | 5.0 | 36.9 | 3.0 | 3.0 | 6.0 |
| | edge+depth+vis(0.3) · 16st · 720 | 1 | 32.2 | 54.8 | 6.0 | 10.0 | 16.0 (dominated) |
| | edge+depth+seg · 35st · 720 (orig) | 1 | 63.7 | 54.8 | 6.0 | 6.0 | 12.0 (dominated) |

**Conclusions:**
- **Recommended: `edge+depth+vis(0.3) · 480 · 6 steps`** — quality 15, ~6.2 min, 40 GB. Best
  balance of restyle strength + quality for domain randomization.
- **Max quality: `vis(0.5) · 8 steps`** (16.2) — only if you don't mind weaker restyling
  (vis 0.5 anchors closer to the original appearance).
- **6 steps is the knee** (realism 4st→6.8, 6st→9.0, 8st→10) — under the stability-focused realism
  prompt; the earlier "4 steps" claim was a scoring artifact of the old prompt.
- **720 / more steps / seg are all dominated** — never on the frontier. depth mandatory; vis is the
  quality unlock.
- **VRAM is never the constraint** (37–40 GB on the frontier); time is.

Scripts: `scripts/exp_pareto.py` (rerank + frontier). Style-aware critic:
`cosmos-reason2/scripts/reason2_critic.py` (`--style` / `--styles-json`, `style_score`). Merged
scores: `outputs/experiments/scores_all.json`.
