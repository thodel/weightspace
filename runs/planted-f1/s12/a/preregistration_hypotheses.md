# Pre-registered hypotheses — planted-f1 / s12 / task (a), budget 10

Programmer: claude-fable-5-1 (Claude Fable 5.1), reasoning effort: default for this harness (not set by me), no extended tools:
only file reading (cat/grep of the pack, the brief, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
`wsp/align.py`; one grep of the `Metrics` dataclass in `wsp/metrics.py` for the field names, disclosed here as a deviation),
file writing, and `py_compile`. No pass was run, no model loaded. Written before the first pass; `program.py` cannot change afterwards.

## Readings that drive the hypotheses
- C0: CER 0.2443; of 3,144 character edits, 1,364 are `e`→`n` and 837 `n`→`e` (70 % of all edits), plus `e`→∅ 71, `n`→∅ 65.
- Diff against the parent: output rows `e` and `n` are the two rows that moved (rel L2 1.19 / 1.10, cosine 0.36 / 0.36),
  all other aligned rows have cosine ≥ 0.88. Norms: parent `e` 7.64 → child 8.33, parent `n` 7.96 → child 7.57, i.e. the child's
  `e` row has the size of a healthy `n` row and vice versa. Everything else about O_18 (blank row rel 0.23, bias Δ +0.11,
  punctuation rows pushed down by 2–3 in bias) looks like ordinary child training.
- Minor oddities: `C_3.co.bias` rel L2 1.09 / cosine 0.68 (but the tensor is tiny, L2 0.10); `L_16.layer.weight_ih_l0` abs max 4.70
  against ≈1.8–2.3 in every other recurrent matrix; the space row has the most negative bias of all (−7.85) and ` `→∅ is the third
  confusion (98).

## Hypotheses (edit class · predicted CER change on the 300 practice lines · conclusion if it fails)

- **H1a (primary): the output rows of `e` and `n` in `O_18.lin.weight` were swapped.** Edit: `set_rows` swapping the two weight
  rows back. Prediction: CER falls by about 0.15–0.19 (0.244 → roughly 0.06–0.09); `e`↔`n` leaves the top confusions.
  If not kept: the rows were not merely swapped (replaced or corrupted) → fall back to H1c.
- **H1b (only if H1a kept): the two biases were swapped with the rows.** Edit: `set_rows` swapping `O_18.lin.bias[e]` and `[n]`.
  Prediction: small, |ΔCER| < 0.005, direction unknown (the two biases are probably within 0.5 of each other). If not kept:
  the planter swapped weights only (or the bias swap is below the resolution of 300 lines); nothing is lost.
- **H1c (only if H1a not kept): the `e`/`n` rows are corrupted beyond a swap.** Edit: `set_rows` transplanting the parent's `e` and
  `n` rows (weight and bias, aligned by character) into the child. Prediction: CER falls by 0.08–0.15 (parent rows are ≈0.2 away
  from healthy child rows, so a partial repair). If not kept: the defect is not in those two rows' values; I stop touching O_18.
- **H2: `C_3.co.bias` was perturbed.** Edit: `transplant` the parent's `C_3.co.bias`. Prediction: |ΔCER| < 0.003, probably not
  kept. If not kept: a tiny bias drifting far in relative terms is natural fine-tuning drift, not a defect.
- **H3: a few entries of `L_16.layer.weight_ih_l0` were blown up (abs max 4.70).** Edit: `add` a delta that returns every entry with
  |w| > 3.0 to the parent's value. Prediction: |ΔCER| < 0.003, probably not kept. If not kept: the outliers are trained weights.
- **H4: the space bias (−7.85) is pushed down and causes the 98 space deletions.** Edit: `add` +1.5 to `O_18.lin.bias[' ']`, run only
  if ` `→∅ is still ≥ 50 in the top confusions after the O_18 repairs. Prediction: −0.002 to −0.006 if true; otherwise CER rises
  slightly (false spaces) and the rule refuses it. If not kept: the space bias is the ordinary CTC/Kurrent behaviour of this hand.

Pass plan (≤ 7 of 10): P1 baseline, P2 H1a, P3 H1b or H1c, P4 H2, P5 H3, P6 H4 (conditional). Every edit is decided by `s.keep`
(paired 95 % interval below 0) plus a check that `m.collapsed` did not rise; nothing kept otherwise. Unspent passes stay unspent.
