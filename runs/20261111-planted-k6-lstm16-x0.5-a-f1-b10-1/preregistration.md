# Pre-registration — 20261111-planted-k6-lstm16-x0.5-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k6-lstm16-x0.5-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s05/a/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# s05 / task (a), budget 10 — pre-registered hypotheses (written before the first pass)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort as set by the harness (not shown to me; default),
tools used: Read + Bash (`cat`, `ls`, `grep` on the pack, this brief and the five allowed `wsp/` files), `py_compile` on
`program.py`. No model loaded, no pass run, no other folder read.

## Reading of the pack

- C0: CER 0.1993, length ratio 0.828, 24 collapsed lines, every top-10 confusion is a deletion (` `, `e`, `n`, `i`, `r`, …).
  Output too short and blank-dominated: the character logits are too weak relative to the blank, not swapped or mis-mapped.
- `L_16` (third BiLSTM) is the outlier block: rel L2 to the parent 0.51–0.55 (L_12/L_14: 0.21–0.30 for the same tensors),
  while the cosines (0.947–0.981) equal those of L_12/L_14 — i.e. the extra distance is a **scale**, not a direction change.
  `L_16.bias_hh` is the smoking gun: cos 0.999 at rel L2 0.505 (a pure ×0.5 would give exactly cos 1, rel 0.5).
  The rms values agree: L_16 weight_ih 0.121 vs L_14 0.217, weight_hh 0.154 vs 0.310, bias_hh 0.42 vs 0.93 — all ≈ ×0.5.
  Solving rel² = (α−1)² + α²δ² with δ from the cosine gives α ≈ 0.47–0.50 for every L_16 tensor. Halving the last LSTM
  halves the O_18 input, so the logits shrink toward the biases; the blank has the largest bias (+1.23) → deletions.
- `C_3.co.bias`: rel L2 1.087, cos 0.68 — large *relative* change, but the tensor is tiny (rms 0.018) and its absolute
  drift (≈0.013 rms) matches C_6/C_9 (0.008–0.012). Probably natural; probed last, cheaply.
- `O_18`: blank row and bias ordinary (bias Δ +0.11 vs parent, rel 0.23 ≈ mean 0.19). The most-moved rows are rare
  punctuation with bias Δ ≈ −2.5 (natural). ` ` has the most negative bias of all (−7.85) — suspicious only if the parent's
  is much higher; checked in-program against the parent (aligned by character) with a robust outlier test.

## Hypotheses (one line each: edit class, predicted direction + magnitude, conclusion on failure)

1. **H1 — L_16 scaled ×0.5.** Edit: `scale` factor 2.0 over all 8 `L_16.layer.*` tensors (bias_ih is zero, no-op).
   Prediction: CER 0.199 → 0.04–0.10 (absolute drop ≥ 0.09), collapsed 24 → ≤ 3, length ratio 0.83 → 0.97–1.02,
   deletions leave the top of the confusion table. If refused: L_16's half-size statistics are natural for this child and
   the deletions have another cause; fall back to H1b.
2. **H1b (only if H1 refused) — milder scale.** `scale` factor 1.5 on the same tensors. Prediction: CER drop 0.02–0.08.
   If refused too: the L_16 block is not scaled; note and move on.
3. **H2 (data-conditional) — swapped output rows.** If a frequent character's `O_18` row (rel L2 to the parent > 0.7)
   matches a *different* parent character's row with cos > 0.85 reciprocally, `set_rows` swaps the two rows (weight+bias).
   Prediction: CER drop ≈ count of that confusion / 12,867 (0.005–0.03). If refused / nothing detected: no row swap.
4. **H3 (data-conditional) — planted bias on a frequent output row.** For aligned characters with ≥ 0.3 % frequency in the
   practice references, bias Δ to the parent with robust z > 4 and |Δ| > 2.5 is flagged; `set_rows` on `O_18.lin.bias`
   puts those biases at parent + median drift. Prediction: CER drop 0.005–0.03 via fewer deletions of those characters.
   If refused / nothing flagged: the biases (incl. ` ` −7.85) are natural.
5. **H4 (conditional) — blank over-biased.** Only if deletions still make ≥ 60 % of the top-10 confusion mass and length
   ratio < 0.97 after H1: `add` −0.6 to `O_18.lin.bias[0]`. Prediction: CER drop 0.002–0.01; at most one repeat if kept.
   If refused: blank calibration is fine once L_16 is repaired.
6. **H5 — C_3.co.bias.** `transplant` of the parent's `C_3.co.bias` (same shape). Prediction: |ΔCER| < 0.003 → the rule
   should refuse it; that refusal confirms the large relative change is natural drift of a tiny tensor, not a defect.
7. **H6 (only if H1 kept) — factor refinement.** `scale` ×1.1 (total 2.2) on L_16. Prediction: |ΔCER| ≤ 0.005, refused;
   refusal means the plant was ×0.5 and factor 2.0 is right. Kept would mean the true factor was ≈ 0.45.

Passes: 1 (C0) + H1 [+ H1b] + H2 [0–1] + H3 [0–1] + H4 [0–2] + H5 + H6 ≤ 10; every edit is guarded by `budget_left ≥ 1`
and decided by `s.keep`. Unspent passes are left unspent. Program ends with `s.freeze()` in all paths.

Expected final state: L_16 ×2 kept, everything else refused; CER ≈ 0.04–0.10.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s05/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
