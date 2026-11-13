# Pre-registration — 20261111-planted-k8-zero-conv-c3-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k8-zero-conv-c3-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s09/b/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 s09, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort (none set explicitly), no extended tools;
read only the three pack files, the brief, and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
`wsp/align.py`. No pass, no model loaded, only `py_compile` on the program.

## Readings

- `c0_practice.md`: CER 1.0000, WER 1.0000, all 300 lines collapsed, length ratio 0.000 — the model emits nothing
  (every reference character → ∅). This is a total collapse, not a confusion pattern.
- `describe.md`: `C_3.co.weight` [32,32,3,13] and `C_3.co.bias` [32] have L2 0.00, rms 0.0000, abs max 0.000 — the whole
  second conv block is exactly zero. Every other tensor has a normal scale (conv rms 0.18–0.19, LSTM weight rms
  0.22–0.31, `bias_hh` rms ~0.9, output rows norm 14.8 ± 4.1).
- `diff_parent_to_subject.md`: `C_3` rel L2 1.0000, cosine nan, 100 % of entries changed (zero vs parent). All other
  blocks sit at rel L2 0.13–0.33 with cosine 0.95–0.999 — ordinary fine-tuning drift from the parent, no second outlier.
  Output rows aligned by character: mean rel change 0.19; the rows that moved most are rare glyphs
  (`«`, `¬`, `_`, `…`, combining marks), which is what a child trained on different data does. New rows have sane norms.
- Mechanism: `C_3` is `Cr3,13,32` (conv + ReLU). With zero weights and zero bias its output is relu(0) = 0 on every frame,
  so `C_6`, `C_9`, the three BiLSTMs and the output layer see a constant input and emit the same (blank-dominated) frame
  everywhere → an empty string on every line. This alone explains CER 1.0 and 300 collapsed lines.

## H1 (applied) — zeroed conv block `C_3`; repair by transplanting the parent's `C_3`

- Hypothesis: `C_3.co.weight` and `C_3.co.bias` were zeroed. The rest of the network is a healthy child of
  `kraken-german-xix-v1`.
- Edit: `transplant` of `C_3.co.weight` and `C_3.co.bias` from the parent (same shapes [32,32,3,13] / [32]; the conv
  stack is index-aligned between parent and child, only the codec grew). One edit, two tensors, decided as a whole.
- Why the parent is the right donor: the child has no information left in `C_3`, so interpolation is meaningless (alpha=1
  is the only choice). The neighbouring blocks moved from the parent by rel L2 0.12–0.33 (C_0 0.13, C_6 0.26), so the
  parent's `C_3` is approximately what the child's `C_3` was; its feature channels are in the same order (the child was
  fine-tuned from the parent, not re-initialised, as the cosines of every other block show).
- Predicted effect: CER falls from 1.000 to roughly 0.10–0.30 on the practice set (parent-level or somewhat worse, since
  `C_6`/`C_9` were co-adapted to the child's own, lost `C_3`); collapsed lines from 300 to near 0; length ratio from 0.0
  to ~0.9–1.0. Direction: large decrease. If it fails (CER stays near 1.0), the collapse has a second cause downstream,
  which the readings do not show.
- Confidence: high. A conv block with all-zero weights and bias cannot be a training outcome; it is a defect, and the
  practice reading is exactly its signature.

## Checked and not acted on

- Output layer: blank row norm 4.64 with bias +1.23 against other rows 14.8 ± 4.1 / −1.9 ± 2.0 — a small, positive-bias
  blank row is normal for a CTC model. Weakest rows are the frequent letters (`e r s t i n a`), strongest the rare
  accented glyphs — normal. Blank row vs parent: rel 0.225, bias Δ +0.11 — ordinary drift.
- Space bias −7.85 is the most negative bias of all rows, among rare glyphs (`ú`, `Ⅲ`, `Ž`, `Ⅱ` at −7.0…−7.3). Odd for the
  most frequent character, but ` ` is not in the "moved most" table and the practice set gives no confusion evidence
  (everything is ∅ because of the collapse). Not confident enough to touch it blind; noted in the program for a
  programmer with passes.
- `L_16.layer.weight_ih_l0` abs max 4.70 vs ~1.8–2.2 elsewhere: one large entry, but the tensor's rel L2 to the parent
  (0.296) and cosine (0.956) are in line with its siblings. Not a defect signature.
- `bias_ih_*` all zero: by construction (brief), also zero in the parent (0/800 changed).
- No LSTM block and no other conv block shows a scale, zero, or sign anomaly. No second edit.

## Program

`program.py`: opens the session, checks that `C_3` is indeed all-zero in the live tensors and that the parent's `C_3` has
the same shapes and a non-zero norm (if either check fails the program applies nothing and freezes), then does one
`hypothesis` → `edit("transplant", ["C_3.co.weight", "C_3.co.bias"], per_tensor=donors)` → `keep_unmeasured` → `freeze`.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s09/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
