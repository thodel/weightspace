# Pre-registration — 20261111-planted-k3-swap-rows-e-n-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k3-swap-rows-e-n-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s12/b/program.py`, copied unchanged.

1. **Subject:** `k3-swap-rows-e-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output rows of `e` and `n` swapped (weights and bias); `runs/planted/plant.json`) — tensor digest `e04ecfb521c6e879805ac6017fbecdaa4545f1e7e315de4804ea722f62b5ba9c`, file sha256 `c0bc09e5c852501c31d1fd20b48a9a2495071d13fa61edaf919573fb89b580e3`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 s12, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort, no extended-thinking toggle
known; tools used: Read / Bash (`ls`, `wc`, `grep`, `cat` on the allowed files, `py_compile`). No pass, no model loaded.
Read: the three pack files, PROGRAMMER_BRIEF.md, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
`wsp/align.py`, `wsp/subject.py` (only the `apply`/`set_tensor` signatures).

## What the readings show

- C0 on the 300 practice lines: CER 0.2443, 3144 edits. The two confusions `e → n` (1364) and `n → e` (837) alone are
  2201 of the 3144 edits (70 %). With `e ≈ 17 %` and `n ≈ 10 %` of German text (≈ 2200 `e`, ≈ 1300 `n` in 12,867
  characters) that is almost every `e` read as `n` and almost every `n` read as `e`; the 1.6 : 1 ratio is the
  frequency ratio of the two letters. The remaining ~940 edits (≈ 7.3 % CER) are the ordinary profile of a Kurrent
  model (space deletions 98, a few `r`/`i`/`.` deletions, 9 collapsed lines).
- Output layer against the parent (rows aligned by character): mean relative change 0.20; the two rows that moved
  most are exactly `e` (rel L2 1.19, cosine 0.356) and `n` (rel L2 1.10, cosine 0.361). Every other "moved" row is a
  rare glyph (`«`, `¬`, `_`, `»`, `…`) with cosine ≥ 0.88 and a bias pushed down by ≈ 2.4 — the usual drift of unseen
  classes in a fine-tuned child, not a defect.
- Norms: parent `e` 7.64 → child `e` 8.33; parent `n` 7.96 → child `n` 7.57. The child's `e` row has the size of a
  (slightly grown) `n` row and vice versa. A row that had merely drifted 20 % like its neighbours would keep cosine
  ≈ 0.95; a row replaced by noise would grow by √2; a row mixed with its neighbour would keep cosine ≈ 0.8.
  Cosine 0.36 with unchanged norm fits one thing: **the `e` and `n` rows of `O_18.lin.weight` have been exchanged.**
  (`e` and `n` are the classic Kurrent look-alikes, so the two healthy rows are moderately similar — cosine 0.36 is
  what a swap of them against the parent should look like.)
- Nothing else stands out as planted. `C_3.co.bias` has rel L2 1.09 / cosine 0.68 against the parent, but it is a
  32-element bias of rms 0.018 (abs max 0.08) behind weights of rms 0.2: noise-level drift of a negligible tensor,
  and neither a scale (cosine would be 1) nor a sign flip (−1). `L_16.layer.weight_ih_l0` has an abs max of 4.7
  (others ≈ 2) but the same rel L2 (0.296) and cosine (0.956) against the parent as every other LSTM input matrix,
  so the outlier is inherited, not planted. Blank row (bias +1.23, Δ +0.11), the space bias (−7.85, the parent's
  too: the row is not among the movers) and the new codec rows (norms 8–15, bias ≈ −0.65) look healthy.

## H1 — the `e` and `n` output rows are swapped (edit class: `set_rows` on the output layer)

- **Edit:** swap rows `cm['e']` and `cm['n']` of `O_18.lin.weight` and the same two entries of `O_18.lin.bias`
  (one `set_rows` edit, `per_tensor` values). The bias is swapped too: a planted class swap most naturally exchanges
  the whole row (weight and bias); the bias readings (Δ −0.04 and +0.18 against the parent) cannot distinguish the
  two cases, and the cost of being wrong on a scalar of that size is a fraction of a percent CER either way.
- **Gate inside the program (no pass needed):** load the parent as donor and compare the child's `e`/`n` rows to the
  parent's `e`/`n` rows by cosine. The swap is applied only if the child's `e` row is closer to the parent's `n` row
  than to the parent's `e` row **and** the child's `n` row is closer to the parent's `e` row than to the parent's
  `n` row (expected: cross cosines ≈ 0.9, straight cosines ≈ 0.36 as in the pack). If the gate fails the program
  edits nothing and notes why.
- **Predicted effect:** CER falls from 0.244 to about 0.07–0.09 (the 2201 `e↔n` substitutions disappear; most of the
  `e → ∅`, `n → ∅`, `∅ → e`, `∅ → n` edits are CTC side-effects of the same swap and should go too). WER from 0.63
  to roughly 0.3. Collapsed lines unchanged or fewer.
- **If it were wrong:** the only way this edit hurts is if the rows were never swapped, in which case the gate
  (cross cosines higher than straight ones) could not pass; the program then ships the model untouched.

## Not applied

- No edit to `C_3.co.bias`, `L_16`, the blank row, the space row or the rare-glyph rows: nothing in the readings
  points to a defect there, and a blind edit on a healthy tensor can only make the model worse.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s12/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
