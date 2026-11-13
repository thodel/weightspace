# Pre-registration — 20261111-planted-k2-negate-row-n-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k2-negate-row-n-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s13/a/program.py`, copied unchanged.

1. **Subject:** `k2-negate-row-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `n` (weights and bias) negated; `runs/planted/plant.json`) — tensor digest `2daff3da912c8f98552bdde696a9c565cfc6eff847652c81a836d5735660d372`, file sha256 `df3b7f92a0e392169f348393a5fdd0c8db10016ef7dad6edd6d6ed02ef347286`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Pre-registration — planted-f1 / s13 / task (a), budget 10

Model: claude-fable-5-1 (Claude Fable 5.1, Claude Code agent). Settings: reasoning effort as configured by the
caller (not visible to me; believed default), no extended tools; tools used: Read/cat/ls of the three pack files and of
the allowed `wsp/` API files, Write for the two deliverables, `py_compile` on `program.py`. No model was loaded, the
program was not run, no pass was spent while writing this. Written before the first pass; `program.py` is fixed from here on.

## Readings that drive the hypotheses

- C0: CER 0.7456, 79 collapsed lines, length ratio 1.61; 7,667 of 9,593 character edits are insertions of `n`
  (plus `e`→`n` 111, `i`→`n` 52, ` `→`n` 48 and `n`→`m`/`r`/∅ when `n` is real).
- Output row `n`: cosine −0.991 to the parent's row, rel L2 2.042 (≈ 2 = a negated vector), norm 7.96 → 8.33,
  bias Δ −0.14. Every other shared row has cosine ≥ 0.88 and rel L2 ≤ 0.63. A negated `n` row makes the `n` logit
  positive exactly where the model wants it negative (every non-`n` frame, including blank frames) — the flood.
- `C_3.co.bias`: rel L2 1.087, cosine 0.679, against 0.26–0.33 / 0.95–0.97 for the other conv biases. But the tensor
  is tiny (rms 0.018, abs max 0.08) in front of a ReLU with pre-activations of order 1, so it can hardly matter.
- Everything else (convs, the three BiLSTMs, the other output rows) sits at the uniform fine-tuning drift of
  rel 0.13–0.33 / cosine ≥ 0.945; the bias extremes (` ` −7.85, rare glyphs ≈ −7) are not compared to the parent in
  the pack, so the program checks them against the donor at run time.

## Hypotheses (one line each: edit class — predicted direction and size on the 300 practice lines — if refused)

- **H1 (sign flip, output row `n`)** — edit class: `set_rows` negating the `n` row of `O_18.lin.weight` (and its bias
  in the same edit if child bias ≈ −parent bias) — predicted: CER falls from 0.7456 to 0.05–0.20 (Δ ≤ −0.55),
  `n` insertions below 200, collapsed lines from 79 to < 5 — if refused: the negation reading is wrong; fallback H1b
  transplants the parent's `n` row and bias (predicted CER 0.10–0.30); if that is also refused the flood comes from
  upstream (LSTM) and I conclude the pack mislocated the defect and stop editing the output layer.
- **H2 (`C_3.co.bias` tampered)** — edit class: `transplant` of the parent's `C_3.co.bias` — predicted: |ΔCER| < 0.01,
  most likely refused (interval contains 0) — if refused: the C_3 bias difference is drift or inconsequential; no loss.
- **H3 (shifted output biases)** — edit class: `set_rows` on `O_18.lin.bias`, for shared rows whose bias Δ to the
  parent lies more than max(3.0, 4·MAD) from the median Δ, set bias = parent bias + median Δ — predicted: no row flagged
  (no edit, no pass) or, if the space row is flagged, CER −0.02 to −0.10 through fewer space deletions — if refused:
  the extreme biases are the parent's own and the fine-tune's rare-class push-down; leave them.
- **H4 (unit-level anomalies elsewhere)** — edit class: `add` (parent minus child) on rows/columns/channels of
  convs, LSTMs or output columns whose relative change exceeds max(0.75, 2.5 × the tensor's median) and on bias
  elements moved by more than 0.5 — predicted: nothing flagged (no edit), else CER −0.01 to −0.05 — if refused:
  the flagged units are tolerated drift; if several tensors were flagged, retry the most severe one alone, once.
- **H5 (residual bias nudge, at most two passes)** — edit class: `add` ±0.5 to `O_18.lin.bias` for any single class
  whose insertions (∅→X) or deletions (X→∅) exceed 1.5 % of the reference characters in the last kept state —
  predicted: −0.002 to −0.01, likely refused — if refused: the residual errors are genuine model errors, not a
  bias defect; stop.

Expected end state: H1 kept, CER ≈ 0.05–0.20; H2–H5 mostly refused or not triggered. Passes: 1 (C0) + 1 (H1)
[+1 H1b] [+1 H2] [+1 H3] [+1–2 H4] [+1–2 H5] ≤ 10; unspent passes are left unspent.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s13/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
