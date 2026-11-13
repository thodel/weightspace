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
