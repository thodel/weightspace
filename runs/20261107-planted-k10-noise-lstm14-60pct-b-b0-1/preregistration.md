# Pre-registration — 20261107-planted-k10-noise-lstm14-60pct-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k10-noise-lstm14-60pct/b/program.py`, copied unchanged.

1. **Subject:** `k10-noise-lstm14-60pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 60 % of the block's; `runs/planted/plant.json`) — tensor digest `930c5c2b0279e885f4e87954711d76aaee3e9d07128fc217156821da237eb2fc`, file sha256 `c55386f265d20e1ba0952391b63ff5e071a8277ec4544cbe9b0a86e9453f2bad`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read/Bash (file reading only), no model loaded, no passes (budget 0).

Readings: CER 0.12, 11 collapsed lines, top confusions are all deletions (space, e, n, r, i ... -> nothing).
Parent->child diff: L_14 weights moved far more than the neighbouring LSTMs (weight_ih rel L2 0.75 / cos 0.79 vs 0.30 for L_12 and L_16;
weight_hh 0.55 vs 0.20; both directions). L_14 weight rms is also inflated (0.26 / 0.34 vs 0.22 / 0.30). Pattern fits additive
noise on the whole of L_14's weight matrices, on top of ordinary fine-tuning drift. Biases of L_14 look normal.
(C_3.co.bias rel 1.09 is a tiny-norm vector, 0.10; not trusted as a defect. Output rows look unremarkable.)

H1: L_14 (weight_ih, weight_hh, both directions) carries added noise. Edit: interpolate those 4 tensors 75 % of the way toward the
parent's L_14 (alpha 0.75). Rationale: child = trained + noise (noise var ~0.48 of parent norm^2 for ih, ~0.26 for hh) while child-parent training
difference is small (~0.09 / ~0.04), so the optimal mix is ~0.85; 0.75 is a conservative choice and cuts expected distance from the
undamaged child by ~5x. Predicted: CER 0.12 -> about 0.06-0.08, fewer collapsed lines, fewer deletions. Risk: parent drift loses
some child-specific learning; still expected net gain. If wrong (L_14 was not noised), CER would rise somewhat.

Not applied: anything on C_3 bias, output layer, other LSTMs (no sign of damage).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
