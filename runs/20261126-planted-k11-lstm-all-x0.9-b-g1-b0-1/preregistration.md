# Pre-registration — 20261126-planted-k11-lstm-all-x0.9-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k11-lstm-all-x0.9-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s04/b/program.py`, copied unchanged.

1. **Subject:** `k11-lstm-all-x0.9.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: all three BiLSTM blocks (weights and biases) x0.9; `runs/planted/plant.json`) — tensor digest `b126e5d8410e8d8d4f17074561f6f94d07f8d67f05c7d686c5f3c9a9ad2a85fd`, file sha256 `4a7b301b0629c376bf631bceba679cc8b7ef7b10a4e85f7c19d8e6a30ec849c1`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
Settings: provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`.

# Hypotheses — s04, task (b) (budget 0, blind programmer)

## Diagnosis: No Defect Present (Healthy Control)

1. **Tensor statistics (`describe.md`):**
   - All 34 tensors across `C_0`, `C_3`, `C_6`, `C_9`, `L_12`, `L_14`, `L_16`, and `O_18` exhibit completely normal L2 norms, RMS values, and abs-max values.
   - In the output layer `O_18`, the CTC blank row (row 0) has norm 4.637 and bias +1.232; non-blank rows have norm 14.812 ± 4.069 and bias -1.921 ± 2.010, with common characters (`e`, `r`, `s`, `t`, `i`, `n`, `a`) having the smallest weight norms (7.57–8.33) as expected for a healthy trained CTC recognizer.

2. **Parent-to-subject diff (`diff_parent_to_subject.md`):**
   - Every convolutional and BiLSTM weight and bias tensor aligns smoothly with the parent model `kraken-german-xix-v1` (weight cosines 0.946–0.993, recurrent `bias_hh` cosines > 0.9989, relative L2 changes 0.11–0.33 across recurrent and conv layers).
   - Output rows aligned across the 225 shared classes show a mean relative L2 change of 0.1930 (blank row relative L2 change 0.2253, bias Δ +0.113). Even the most-changed rows (`[`, `«`, `_`, `»`, `¬`) retain high cosine similarity (0.862–0.950) and normal norms, with no swapped, zeroed, sign-flipped, or spiked rows.

3. **Practice-set readings (`c0_practice.md`):**
   - Baseline CER is 0.0727 (WER 0.1903, length ratio 0.959, 12 collapsed lines, 935 edits across 12,867 reference characters), which is the healthy baseline performance of this fine-tuned child model.
   - The top-10 confusions are ordinary deletions of the most frequent characters (` `, `e`, `n`, `r`, `i`, `s`, `.`, `o`, `u`) and space insertions (`∅ → ' '`), with no anomalous substitution or deletion spike.

## Planned Action

- **Hypothesis:** The subject model contains no planted defect. Any blind weight edit risks degrading a healthy model.
- **Edit:** None (no-op).
- **Predicted effect:** ΔCER = 0.0000 (preserves healthy baseline CER of 0.0727).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s04/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 119242 input / 1401 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
