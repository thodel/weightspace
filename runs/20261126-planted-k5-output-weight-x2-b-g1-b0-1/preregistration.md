# Pre-registration — 20261126-planted-k5-output-weight-x2-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k5-output-weight-x2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s14/b/program.py`, copied unchanged.

1. **Subject:** `k5-output-weight-x2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output layer weight matrix x2 (bias unchanged); `runs/planted/plant.json`) — tensor digest `2985c3e3f813d9b43bc0dfc951852ac29bb071c03729a7ce67a3a03235ab5fbe`, file sha256 `0e7be185c0f5730a826662ff3087b16b829eba160c7d6bb82170133d0957060b`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer, provider default reasoning settings)
Tools used: `read_file`, `write_file`, `finish_round`

# Hypotheses — s14 task (b) (budget 0)

## Diagnosis: Healthy Model (No Defect Detected)

- **Hypothesis H0 (no defect / control subject):** All readings in `describe.md`, `diff_parent_to_subject.md`, and `c0_practice.md` are consistent with an unmodified, healthy fine-tuned child model:
  1. **Practice-set performance (`c0_practice.md`):** Baseline CER is `0.0650` (WER `0.1751`, 836 character edits across 12,867 reference characters, length ratio `0.979`, 8 collapsed lines). The top-10 confusions are ordinary minor CTC deletions of common characters (` ` → `∅`: 78, `e` → `∅`: 34, `n` → `∅`: 28, `i` → `∅`: 21, `r` → `∅`: 19, `s` → `∅`: 16) and space insertion (`∅` → ` `: 25). There are no abnormal substitution spikes or systematic character losses.
  2. **Tensor statistics (`describe.md`):** Every convolutional layer (`C_0`, `C_3`, `C_6`, `C_9`), bidirectional LSTM layer (`L_12`, `L_14`, `L_16`), and output layer (`O_18.lin.weight` L2 `528.65`, `O_18.lin.bias` L2 `47.85`) exhibits normal L2 norms, RMS values, and maximum absolute entries. In `O_18`, the blank row norm (`9.274`) and bias (`+1.232`) and non-blank row norms (`29.625 ± 8.138`) and biases (`-1.921 ± 2.010`, max non-blank bias `s` at `+0.40`) show no outliers, zeroed rows, or bias spikes.
  3. **Parent-to-subject diff (`diff_parent_to_subject.md`):** All weight tensors have high cosine similarity with the parent (`0.946` to `0.999`), and the output rows that moved most retain high cosine similarity (`0.910` to `0.998`) with smooth norm growth from fine-tuning.
- **Edit:** None (0 edits applied).
- **Predicted effect:** CER remains at `0.0650` (`ΔCER = 0.0000`), avoiding degradation on a healthy model.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s14/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (4 API calls, 38546 input / 1127 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
