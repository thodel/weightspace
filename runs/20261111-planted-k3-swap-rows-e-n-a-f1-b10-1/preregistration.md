# Pre-registration — 20261111-planted-k3-swap-rows-e-n-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k3-swap-rows-e-n-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s12/a/program.py`, copied unchanged.

1. **Subject:** `k3-swap-rows-e-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output rows of `e` and `n` swapped (weights and bias); `runs/planted/plant.json`) — tensor digest `e04ecfb521c6e879805ac6017fbecdaa4545f1e7e315de4804ea722f62b5ba9c`, file sha256 `c0bc09e5c852501c31d1fd20b48a9a2495071d13fa61edaf919573fb89b580e3`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s12/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
