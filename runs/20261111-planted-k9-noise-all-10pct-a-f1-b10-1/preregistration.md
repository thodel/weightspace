# Pre-registration — 20261111-planted-k9-noise-all-10pct-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k9-noise-all-10pct-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s10/a/program.py`, copied unchanged.

1. **Subject:** `k9-noise-all-10pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: Gaussian noise on every weight tensor, 10 % of each tensor's norm; `runs/planted/plant.json`) — tensor digest `fce9af3f1f4418c4fae94ad695de17f982c7f128c3e84f7de67108e4fd545690`, file sha256 `299cfd10c352e26617380b0dc2a656233b1576497e5b65cc072018ce6a75426d`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Pre-registered hypotheses — planted-f1 / s10 / task (a), budget 10

Model: claude-fable-5-1 (Claude Fable 5.1, Claude Code agent). Settings: default reasoning effort, tools used: Read/Bash (cat, ls) on the
three pack files and the five allowed `wsp/` files, Write, `py_compile`. No pass run, no model loaded.

## What the readings say

- C0: CER 0.5509, WER 0.7653, 123 of 300 lines collapsed, length ratio 0.554. The top-10 confusions are **all deletions**
  (` `→∅ 1001, e 630, n 454, i 363, r 316, s 265, t 232, a 229, u 194, h 167): the model emits too little, and what it emits is
  mostly right. That is a blank-dominance / weak-feature pattern, not a codec or row-swap pattern (no substitutions).
- Output layer: blank bias +1.23, every other row's bias negative (−1.92 ± 2.01). The **space bias is −7.85, the most negative of
  all 297 rows** although space is the most frequent character; the other extreme biases (`ú`, `Ⅲ`, `Ž`, `Ⅱ`) are rare characters.
  Frequent letters have the weakest rows (e 7.6 … n 8.5 vs 14.9 ± 4.1 for the rest) — probably natural (rare rows grow large),
  but a planted row scaling on frequent letters would look exactly like this and produce exactly these deletions.
- Parent → subject: all block changes look like ordinary fine-tuning (rel L2 0.15–0.34 with cos ≈ 1/√(1+r²)). The one apparent
  outlier, `C_3.co.bias` (rel 1.087, cos 0.679), is an absolute change of ≈0.07 — the same size as the other conv biases' changes;
  it is large only relative to the parent's tiny C_3 bias. I do not read it as a defect.
- `L_16.layer.weight_ih_l0`: abs max 4.73 (every other tensor ≤ 2.9), rms 7 % above its reverse twin. A sprinkle of planted
  large entries (~100 at ±4.7) would produce exactly these two numbers while leaving rel L2 to the parent (0.313) unremarkable.

## Hypotheses (tested in this order; each conditional one costs a pass only if its trigger fires at run time)

- **H1 — sparse/structural damage in a hidden tensor** (edit class: *set entries to donor / rescale rows-cols to the donor's norm*).
  At run time compare every conv/LSTM tensor with the parent: entries whose fine-tuning delta is > 12 robust σ and > 0.5 absolute,
  and rows/columns (out/in channels, gate rows, input features) whose child/parent norm ratio is < 0.5× or > 2× the tensor's median
  ratio, are reset (entries → parent value; zeroed rows → parent row; scaled rows → rescaled). One combined `add` edit.
  Prediction if it fires and is the main defect: CER 0.55 → below 0.25 (Δ ≤ −0.30), collapsed lines 123 → < 30. If the combined edit
  is refused and more than one tensor was touched, retry the single most-changed tensor alone. If it fails: the hidden anomaly
  (L_16 abs max) is not the cause; the defect is in the output layer or is not repairable from the parent.
- **H2 — output-layer weight rows/columns altered** (edit class: *rescale rows / columns, reset sparse entries to the donor*, aligned by
  character). Rows with norm ratio < 0.62× or > 1.6× the median ratio over the 225 shared characters, columns (features) that are
  dead or scaled over the shared rows, and > 12 σ sparse entries. Prediction if it fires: Δ −0.05 to −0.30 (if the weak frequent rows
  are a planted scaling, the e/n/i/r/s/t/a deletions should drop by half). If it fails: the weak frequent rows are natural.
- **H3 — output biases planted (space first)** (edit class: *bias set*). Bias delta to the parent per shared character; reference
  = median delta over the frequent lowercase letters (μ_F). Space/punctuation/frequent letters whose delta deviates by more than
  max(4σ_F, 2.0) (uppercase: 3.5; other characters: only > +4.0) are set to parent bias + μ_F. Prediction if space fires: Δ −0.03 to
  −0.08 (1001 space deletions = 7.8 % of the reference characters, part of them inside collapsed lines), WER down more. If it
  fails: the space bias is the model's own calibration and spaces are lost by the collapse, not by the bias.
- **H4 — blank dominance (symptom repair)** (edit class: *bias add on row 0*). Blank bias −1.0; if kept and lines still collapse
  (collapsed > 5 or deletions > 2× insertions) another −1.5, then −2.0 (cumulative −4.5 at most). If the first −1.0 is refused and
  collapsed > 40, one big-step test of −3.0. Prediction: first step Δ −0.05 to −0.20, collapsed 123 → < 60; later steps smaller.
  If refused: the collapse is not marginal (features dead, not merely out-voted), so no output-bias repair can fix it.
- **H5 — blank row weights altered** (rel change 0.44, the largest of the output rows; edit class: *set_rows from donor*). Row 0 of
  `O_18.lin.weight` ← parent row 0. Prediction: Δ −0.05 to −0.20 if planted; else refused (parent row misaligned to fine-tuned
  features). If refused: the blank row's change is fine-tuning.
- **H6 — the `C_3.co.bias` anomaly is NOT a defect** (control; edit class: *transplant from donor*). Only if a pass is left.
  Prediction: |Δ| < 0.01, refused by the rule. If it is kept with a clear gain, my reading of the relative-change table was wrong
  and absolute change size is not a safe criterion.

Budget: pass 1 = C0 baseline; at most 9 further passes, one per edit, each decided by `s.keep` (paired 95 % interval) plus my own
checks on `m.cer`, `m.collapsed`, `m.confusions`. Unspent passes are left unspent; the program ends with `s.freeze()`.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s10/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
