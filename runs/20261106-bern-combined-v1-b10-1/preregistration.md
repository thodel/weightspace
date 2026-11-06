# Pre-registration — 20261106-bern-combined-v1-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 06.11.2026 on asteraix. Keep rule of PROTOCOL §3 as of 06.11.2026 (paired 95 % interval against the last kept state must lie below 0, `min_effect` = 0), enforced by `Session.keep()`; the programmer also reverts any edit that raises the collapsed count.

1. **Subject:** `models/kraken/kraken-bern-combined-v1.mlmodel` — tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`. Trained from the 25.10 export, in which 44 of the 50 Hertwig pages carry another page's image (docs/HERTWIG_REVISIONS.md).
   Donors, read-only: `kraken-bern-hertwig-v2` (the parent; tensor digest `f216564e…`, file sha256 `479392ab7899b831b3c3872dc0f178a894495757b2df2047f2b695c4a5dcc8f3`) and `kraken-bern-schlussredaktion-v1` (the clean sibling: same parent, clean Schlussredaktion pages of 24.10; tensor digest `a0a8e336…`, file sha256 `d26eb02dfaf89f8f222165a1e5c982e38f19a915a5a744e4ad6c15f69b71f526`). Output rows aligned by character.
2. **Practice set:** `data/practice/bern-train-300.jsonl` — 300 lines (103 Hertwig, 197 Schlussredaktion), list sha256 `474414e3be95787da5f08f4bdcfc04adcfc024d8518b72cab8e885367977343f`. Train side of both projects combined-v1 was trained on, with the **correct** 22.10 Hertwig images. Audit clean (`data/practice/overlap.json`). A Hertwig-only set would reward turning the subject back into hertwig-v2, which was trained on those very lines.
3. **Held-out set:** `data/heldout/bern-eval.jsonl` — 545 lines, the 6 Hertwig test pages of the 22.10 export and the 8 Schlussredaktion test pages, list sha256 `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`. Audit (`data/heldout/overlap-bern-eval.json`): all rows from `_test` projects, no page in any practice list or among the 168 training pages.
4. **Budget:** 10 passes. Pass 1 is C0.
5. **Starting point** (C0/C1 before the session, `docs/baselines/bern-train-300.json`, `docs/baselines/heldout-bern-eval.json`):

   | CER | practice Hertwig | practice Schlussred. | practice all | held-out Hertwig | held-out Schlussred. | held-out all |
   |---|---:|---:|---:|---:|---:|---:|
   | combined-v1 (subject) | 0.205 | 0.118 | 0.146 | 0.267 | 0.135 | **0.193** |
   | hertwig-v2 (parent) | 0.114 | 0.299 | 0.241 | 0.236 | 0.306 | 0.275 |
   | schlussredaktion-v1 (sibling) | 0.236 | 0.113 | 0.151 | 0.295 | 0.131 | 0.204 |

   The subject is the best of the three overall; its noise damage is about 3 points on Hertwig held-out lines (9 on Hertwig practice lines, which the parent was trained on). The sibling is worse than the subject on Hertwig. The premise "combined-v1 is worse than its parent" (docs/MODELS.md, 28.8 % vs 16.7 %) does not hold on correctly paired pages.

6. **Hypotheses** (predicted change of practice CER against the last kept state; edit class):
   - H1: blank bias back to hertwig-v2's value (0.998 → 0.879, the readings' first hypothesis). Predicted −0.2 to +0.2 points; most likely refused (phase 1 found blank edits noise-sized). `add` on `O_18.lin.bias[0]`.
   - H2–H6: one block at a time from the clean sibling: output layer (rows aligned, `ƒ` kept), L_16, L_14, L_12, conv stack. Predicted: each **refused**, with Hertwig lines worse by 1–3 points and Schlussredaktion lines unchanged within ±0.5, because the sibling reads Hertwig worse than the subject. A kept transplant would mean the noise damage is concentrated in that block. `transplant`.
   - H7 (only if H2 is not kept): output layer from hertwig-v2 (the readings' hypothesis (b)). Predicted net **worse**: Hertwig −2 to −5, Schlussredaktion +3 to +8; refused.
   - H8: L_16 half-way back toward hertwig-v2 (`interpolate`, α = 0.5). Predicted practice CER within [−0.5, +1.0]; Hertwig −1 to −3, Schlussredaktion +0.5 to +2; most likely refused.
   Per-project numbers are read from `readings/pass_*.jsonl` after the session, outside the sandbox.
7. **Controls:** C0 (pass 1), C1 parent and sibling on practice and held-out (table above), C6 blind = the readings (docs/READINGS.md). C4 (retrain on the corrected data) is a trainer job and needs an explicit OK; not run here. C5 sham only if a gain is reported.
   Programmer: Claude Opus 5.5 (claude-opus-5-5) in Claude Code on asteraix; `program.py` was written in full before the sandbox run; the sandbox command line is in `summary.json`.
