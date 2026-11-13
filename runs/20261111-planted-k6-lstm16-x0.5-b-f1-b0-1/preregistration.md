# Pre-registration — 20261111-planted-k6-lstm16-x0.5-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k6-lstm16-x0.5-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s05/b/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 / s05 / task (b), budget 0 (blind programmer)

- Model: claude-fable-5-1 (Claude Fable 5.1). Reasoning effort: default (medium-low). Tools: Read/Bash (cat, grep, ls)
  on the three pack files, the brief and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
  `wsp/align.py`; `py_compile` of `program.py`. No model loaded, no pass run, sibling `a/` not read.

## What the readings say

- C0: CER 0.199, WER 0.491, 24 of 300 lines collapsed, length ratio 0.828. Every one of the top-10 confusions is a
  deletion (` `, `e`, `n`, `i`, `r`, `c`, `s`, `u`, `t`, `m` → ∅), i.e. the model under-emits across the board: the blank
  wins too often. Nothing character-specific.
- Diff against the parent: blocks C_0/C_6/C_9/L_12/L_14 moved like an ordinary fine-tune (rel L2 0.13–0.31, cosine
  0.95–0.99). **L_16 (third BiLSTM) stands out**: every non-zero tensor of it has rel L2 0.505–0.550, and the two
  `bias_hh` vectors have rel L2 0.5054 / 0.5059 with cosine 0.99897 / 0.99907. A vector that moved by half its length
  while keeping cosine 0.999 is the parent vector *times ≈0.5*, not a fine-tuned one (L_12/L_14 `bias_hh` moved by
  0.03–0.04 with the same cosine). Solving `child = k·parent + drift` from rel and cos gives k ≈ 0.495 for both biases.
  The weights fit the same story: a 0.5 scaling of a parent-plus-ordinary-drift tensor predicts rel ≈ 0.52 and
  cos ≈ 0.96, observed 0.51–0.55 / 0.947–0.981.
- Absolute norms agree: in the subject, L_16 is about half of L_14 (`weight_ih` rms 0.121 vs 0.217, `weight_hh` L2 61.8 vs
  123.9, `bias_hh` L2 12.0 vs 26.3) although the two layers have identical shapes and the first two LSTMs match each other.
- Mechanism → symptom: halving the gate pre-activations of the last LSTM squashes its output (gates toward 0.5,
  `tanh` candidates shrunk), so the logits `W·h + b` lose contrast and the output biases dominate; the blank bias is +1.23
  while letters sit around −1.9 → blank wins → deletions, short lines, whole lines collapsing. Exactly what C0 shows.

## H1 (applied, guarded): L_16 was scaled by ≈0.5 — undo it

- Edit class: `scale` of the six non-zero L_16 tensors (`weight_ih_l0`, `weight_hh_l0`, `bias_hh_l0`, forward and reverse)
  by one common factor 1/median(k_i), where k_i = ⟨child_i, parent_i⟩ / ⟨parent_i, parent_i⟩ is the least-squares scale of
  each tensor against the parent (expected ≈ 0.5, so the factor ≈ 2.0). `bias_ih` is all-zero and untouched.
- Guard (the program checks before editing, using only the parent as a donor; no pass): every k_i in [0.35, 0.65] and every
  cosine ≥ 0.9, while the same statistics for L_14 give k in [0.8, 1.25]. If the guard fails, nothing is applied and the
  session is frozen unchanged — a healthy child whose last LSTM is legitimately small must not be doubled.
- Predicted effect: CER falls from 0.199 to roughly 0.05–0.10 (at least −0.08 absolute); the deletion confusions shrink by
  most of their count, collapsed lines → ≈0, length ratio → ≈1.0. If it fails (CER unchanged or higher) the L_16 norm ratio
  is a property of this child's training rather than a defect, and the deletions have another cause (output layer /
  blank calibration).

## Considered and not applied

- Space bias −7.85 (most negative of all 297 rows) with 301 space deletions. But spaces are deleted at about the same rate
  as `e` (251), so this is the general under-emission, not a space-specific bias defect; space is not among the rows that
  moved most against the parent. Changing it blind would mis-calibrate a healthy row. Not touched.
- Blank row: bias Δ +0.11 vs parent, rel L2 0.23 (block mean 0.19) — ordinary. Not touched.
- `C_3.co.bias`: rel L2 1.09, cosine 0.68, but its norm is 0.10 (rms 0.018, parent's ≈ 0.07); a tiny bias of a ReLU conv
  whose filters have norm ≈ 7. Even if noise was added here its effect is negligible, and a transplant could undo a real
  fine-tune. Not touched.
- Rows that moved most (`«`, `¬`, `_`, `…`, `[`, …) are rare characters with bias Δ ≈ −2.5; consistent with fine-tuning
  on a hand where they do not occur. Not touched.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s05/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
