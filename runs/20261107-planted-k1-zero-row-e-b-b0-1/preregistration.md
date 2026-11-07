# Pre-registration — 20261107-planted-k1-zero-row-e-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k1-zero-row-e/b/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Blind programmer (b), budget 0
Model: claude-sonnet-5-5. Reasoning effort: low/default. Tools: Read, Bash (cat, ls, grep, py_compile only); no model loaded, no passes.

## H1 (apply): the output row of `e` was zeroed
Evidence: describe.md gives `e` row norm 0.00 (every other row >= 7.7); diff shows `e` rel L2 1.000, cosine 0.000, norm 7.64 -> 0.00;
practice CER 0.162 with `e` -> nothing 910x and `e` -> a/i/o/r/c/u as top confusions (the most frequent German letter is never emitted; with a zero row
only the bias, -0.14, competes with the other classes). Nothing else is anomalous: other moved rows are ordinary fine-tuning drift, length ratio 0.90 matches deleted `e`s.
Edit: set_rows on `O_18.lin.weight`, row cm['e'], values = parent's row pm['e'] (parent 7.64 norm, same scale as the child's common-letter rows 7.7-8.3). The child bias for `e` is kept (-0.142 vs parent, not anomalous).
Predicted effect: large improvement, CER 0.162 -> roughly 0.05-0.09 (the parent basis has drifted ~20 % in the LSTM, so not a perfect row).
If wrong (e.g. the child was always like this): the edit costs little, since a zero row cannot be better than a trained-ish row.
Guard in program: applied only if the current `e` row norm < 1e-6.

## Not applied
- C_3 bias rel L2 1.09 (cos 0.68) in the parent diff: C_3 bias is tiny (rms 0.018), likely ordinary fine-tuning; no evidence of fault.
- `L_16.weight_ih_l0` abs max 4.7 and bias extremes: no clear defect; not touching.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
