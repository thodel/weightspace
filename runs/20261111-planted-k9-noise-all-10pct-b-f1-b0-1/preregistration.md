# Pre-registration — 20261111-planted-k9-noise-all-10pct-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k9-noise-all-10pct-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s10/b/program.py`, copied unchanged.

1. **Subject:** `k9-noise-all-10pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: Gaussian noise on every weight tensor, 10 % of each tensor's norm; `runs/planted/plant.json`) — tensor digest `fce9af3f1f4418c4fae94ad695de17f982c7f128c3e84f7de67108e4fd545690`, file sha256 `299cfd10c352e26617380b0dc2a656233b1576497e5b65cc072018ce6a75426d`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 s10, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort as configured by the harness (not visible to me;
default). Tools used: Read (the three pack files, the brief, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`,
`wsp/kraken_subject.py`, `wsp/align.py`), Bash (`ls`, `wc`, `py_compile`), Write. No pass, no model loaded, nothing else read.

## What the readings say

- C0: CER 0.5509, WER 0.7653, **length ratio 0.554** (≈ the CER: the errors are almost all deletions), 123 of 300 lines
  collapsed. The top-10 confusions are all `x → ∅`, in German letter-frequency order (e, n, i, r, s, t, a, u, h). Deletion
  rates of the frequent letters are ~30–38 % (e 630 of ≈2000) — roughly the collapsed-line share — while the space is
  deleted ~50 % of the time (1001 of ≈2000): the space has an extra, class-specific problem on top of the general collapse.
- Output layer: the **space row's bias is −7.85**, the lowest of all 297 rows, 3σ below the non-blank mean (−1.92 ± 2.01),
  lower than the rare characters `ú` −7.33, `Ⅲ` −7.12, `Ž` −7.12, `Ⅱ` −7.01, while the frequent letters sit near 0 (`s` +0.40,
  `d` +0.29). A frequent class with the most negative prior of the whole codec is not a training outcome. The "moved most"
  table of the diff is selected on the weight rows, so a bias-only shift would not appear there.
- Blank row: bias +1.23, Δ vs parent +0.11 (natural); weight rel change 0.44 with unknown cosine — a planted scale of the
  child's own row is indistinguishable from drift without a pass. Not touched.
- `C_3.co.bias`: rel L2 1.087, cos 0.679 — the only block whose bias moved ~5× more than its weights (the other three move
  together). But the tensor's norm is 0.10 (parent ≈ 0.07), i.e. a change of ~0.013 per channel against pre-activations of
  order 1: functionally negligible whether planted or not. Not touched (a transplant would be a no-op coin flip).
- `L_16.layer.weight_ih_l0`: abs max 4.73 (every other LSTM weight ≤ 2.86), L2 137.5 vs 128.7 for its reverse twin (L_12
  and L_14 twins agree within 1 %). Its rel L2 to the parent (0.313) is normal, so at most a handful of entries can be
  planted; handled by the spike scan below.
- Block-level: every weight tensor has rel L2 0.15–0.34 and cos 0.94–0.99 to the parent — no whole-tensor scale, zeroing or
  noise is visible anywhere. `bias_ih` tensors are zero by construction.

## Hypotheses and edits (each applied only if its guard fires; otherwise a note, no edit)

**H1 — output-bias shift on a frequent class (the space).** Edit class: `set_rows` on `O_18.lin.bias`.
Guard, for every character shared with the parent: deviation of (child bias − parent bias) from the median drift of the
lowercase rows a–z äöüß is < −4.0 **and** the parent bias is > −3.5 (a normal class in the parent) **and** the child bias
is ≤ −6.0; or the deviation is > +4.0 and the child bias ≥ +2.0. Natural drift observed in the diff is −2.7 … +0.7, so
only a planted shift can pass. Expected hit: the space only. Repair: bias ← parent bias + median lowercase drift (the best
blind estimate of the child's natural value; the parent's bias alone would ignore the fine-tuning drift).
Prediction: CER **down by 0.01–0.04** (0.55 → 0.51–0.54) from spaces recovered on the non-collapsed lines; ~200–400 of the
1001 space deletions; collapsed-line count unchanged unless "collapsed" counts spaceless output. If the parent's space bias
is also ≈ −7.8 the guard does not fire, nothing is applied, and the −7.85 is a (strange) natural value — predicted change 0.
If applied and the CER rises, the space row's weights were adjusted to the low bias, which no reading shows.

**H2 — planted spikes in conv/LSTM tensors.** Edit class: `add` (restore the flagged entries to the parent's values).
Guard: any entry with |child − parent| > 1.5 in a conv or LSTM tensor (typical per-entry change there is 0.03–0.08, so 1.5
is > 20× the natural change; the output layer is excluded because its rows drift far more). Prediction: if it fires (most
likely in `L_16.layer.weight_ih_l0`), CER **down by 0 to 0.01**; if the 4.73 is also in the parent nothing fires. If it
fires on many entries of one tensor, the readings' block-level statistics were misread.

**H3 — zeroed or non-finite rows/columns/channels.** Edit class: `set_rows` / `set_cols` from the parent.
Guard: a row (out-channel) or column (in-channel) that is exactly zero in the child and not in the parent, or any non-finite
entry. Exact zero rows never arise from training. Nothing in the readings suggests one, so the predicted outcome is **no
edit**; if it fires, the restored rows carry the parent's values without the fine-tuning drift (partial repair, CER down).

## Not applied, and why

- The general under-emission (123 collapsed lines, ~30 % deletion of every letter) is the dominant defect but no reading
  localises it. Lowering the blank bias or shrinking the blank row blind would trade deletions for insertions on the 177
  good lines; without a pass that is a gamble, not a repair. Needs task (a).
- `C_3.co.bias`: see above — negligible magnitude either way.
- No interpolation towards the parent: its CER on this hand is much worse than a healthy child's.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s10/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
