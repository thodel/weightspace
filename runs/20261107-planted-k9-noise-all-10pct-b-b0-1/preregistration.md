# Pre-registration — 20261107-planted-k9-noise-all-10pct-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k9-noise-all-10pct/b/program.py`, copied unchanged.

1. **Subject:** `k9-noise-all-10pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: Gaussian noise on every weight tensor, 10 % of each tensor's norm; `runs/planted/plant.json`) — tensor digest `fce9af3f1f4418c4fae94ad695de17f982c7f128c3e84f7de67108e4fd545690`, file sha256 `299cfd10c352e26617380b0dc2a656233b1576497e5b65cc072018ce6a75426d`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Blind programmer (b), budget 0
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools used: Read/Bash (read pack files and brief only, no model loaded, no passes).

Readings: CER 0.551, 123/300 lines collapsed, length ratio 0.554, top confusions are all "character -> nothing" (deletions, spaces 1001, e 630, n 454...).
The model under-emits: the blank wins too often.

## H1: blank dominates (blank bias +1.23 vs others -1.9; blank row norm 5.0)
Possible edit: lower O_18.lin.bias[0] by a constant. Predicted CER down if true. Not applied: the blank's high bias and low row norm are typical of CTC
models, the parent comparison shows only +0.113 bias drift, and I cannot size the shift without a pass.

## H2: C_3.co.bias changed 109 % vs parent (cos 0.68) while its weights changed 19 %
Odd but tiny tensor (L2 0.10); no donor-vs-child evidence of which is right; a transplant could break the fine-tuned conv stack. Not applied.

## H3: broad diffuse change (every tensor moves 15-30 % from parent, no single locus)
Consistent with ordinary fine-tuning (72 new characters) or with diffuse noise; no tensor shows a localised anomaly (no zero rows, no extreme norms,
L_16.weight_ih max 4.7 is mildly high only). No targeted repair is identifiable.

## Decision
No edit I am fairly sure about; apply nothing, freeze. Suspicion: diffuse weight perturbation across all tensors (not repairable blind), or a mild blank over-dominance.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
