# Pre-registration — 20261111-planted-k11-lstm-all-x0.9-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k11-lstm-all-x0.9-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s04/b/program.py`, copied unchanged.

1. **Subject:** `k11-lstm-all-x0.9.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: all three BiLSTM blocks (weights and biases) x0.9; `runs/planted/plant.json`) — tensor digest `b126e5d8410e8d8d4f17074561f6f94d07f8d67f05c7d686c5f3c9a9ad2a85fd`, file sha256 `4a7b301b0629c376bf631bceba679cc8b7ef7b10a4e85f7c19d8e6a30ec849c1`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 / s04 / task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort low-medium (default for this
subagent), tools used: Read/Bash (cat, ls, grep, py_compile) on the pack, the brief and the allowed `wsp/` API
files only. No pass, no model loaded, program not run.

## What the readings show

1. **Space bias is an extreme outlier.** `O_18.lin.bias[' '] = -7.85` is the most negative bias of all 297
   classes, below even the rarest glyphs (`ú` -7.33, `Ⅲ` -7.12, `Ⅱ` -7.01) and ~3 sd below the layer mean
   (-1.92 ± 2.01). Space is the most frequent class in the training text; frequent classes in this layer sit near
   zero or slightly positive (`s` +0.40, `E` +0.33, `d` +0.29) and have the smallest row norms (e, r, s, t, i, n, a
   at 7.6–8.3). A trained space row would never end at -7.85. Consistent with it, the top confusion of C0 is
   ` ` → ∅ (124 of 935 edits, 2.3× the next confusion `e` → ∅ 55), and the length ratio is 0.959 (hypothesis
   too short). The space row is not among the "moved most" rows (that table ranks by weight-row rel L2), so the
   weight row of the space is probably untouched and the bias alone was pushed down.
2. **Everything else looks like ordinary fine-tuning drift.** Conv and LSTM blocks have rel L2 0.13–0.33 and
   cosine 0.94–0.99 to the parent, uniformly across blocks and directions; `bias_ih` all zero by construction.
   `C_3.co.bias` has rel L2 1.09 / cos 0.68, but the tensor is tiny (L2 0.10, rms 0.018; parent ≈ 0.07) so it
   is relative noise, not a defect worth editing. `L_16.layer.weight_ih_l0` has an abs max of 4.23 (other
   `weight_ih` tensors max at 1.2–2.0) with otherwise normal L2/rel L2/cosine; a possible single-element spike,
   not visible in the aggregate statistics. Blank row (norm 4.64, bias +1.23, Δ +0.11 to parent) is normal.
   12 collapsed lines of 300 cannot be attributed to any tensor from these readings; I leave them alone.

## Hypotheses, edits, predictions

| # | hypothesis | edit | predicted effect on practice CER |
|---|---|---|---|
| H1 | The bias of the space row was pushed to -7.85 (planted). Restoring it to where the parent (plus the frequent-letter drift of this child) puts it removes most of the 124 space deletions. | `set_rows` on `O_18.lin.bias`, index `cm[' ']`, value = parent bias of `' '` + median(child − parent bias over frequent Latin letters). **Guarded:** applied only if the child-minus-parent bias Δ of the space is below -3 and is a >5-MAD outlier against the Δ of the frequent letters (so an inherited, naturally low value is not touched). Fallback without parent: median bias of the frequent letters, same child-only outlier guard. | CER 0.0727 → about 0.065–0.068 (−0.5 to −0.8 pp): roughly 80–110 fewer space deletions, perhaps 10–30 new space insertions. WER should fall more (−2 to −4 pp) since every missing space merges two words. |
| H2 | Isolated weight elements far from the parent (|child − parent| > 1.5, i.e. >20× the per-element drift rms of ~0.06) in conv/LSTM weight or bias tensors are planted spikes (candidate: the 4.23 in `L_16.layer.weight_ih_l0`). | `add` of (parent − child) on exactly those elements, one edit per affected tensor. **Guarded:** only tensors with 1–200 such elements; 0 → no edit, >200 → not a spike, skipped with a note. Output layer excluded (its elements are legitimately large and rows drift more). | If a spike exists: small improvement, −0.1 to −0.5 pp, possibly fewer collapsed lines. If none: no edit written. |

Not applied: anything on `C_3.co.bias`, the blank row, the LSTM gates, or the conv kernels as wholes — the
readings do not show a defect there, and a wrong edit at budget 0 cannot be caught.

## What the program does

Loads the parent as donor, aligns output rows by character, evaluates the guards above on the weights alone
(no forward pass), and for every guard that fires writes `s.hypothesis` → `s.edit` → `s.keep_unmeasured`;
every guard that does not fire is recorded with `s.note`. Ends with `s.freeze()`.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s04/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
