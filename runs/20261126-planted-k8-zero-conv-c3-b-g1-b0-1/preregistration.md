# Pre-registration — 20261126-planted-k8-zero-conv-c3-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k8-zero-conv-c3-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s09/b/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — s09 (task b, budget 0)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis from pack readings

1. **`describe.md`**:
   - Both tensors of the second convolutional layer `C_3` (`C_3.co.weight` of shape `[32, 32, 3, 13]` and `C_3.co.bias` of shape `[32]`) have `L2 = 0.00`, `rms = 0.0000`, `abs max = 0.000`. They are completely zeroed out.
   - All other convolutional blocks (`C_0`, `C_6`, `C_9`), BiLSTM blocks (`L_12`, `L_14`, `L_16`), and the output layer (`O_18`) have healthy norms and distributions.
2. **`diff_parent_to_subject.md`**:
   - Compared against parent `kraken-german-xix-v1`, block `C_3` has `rel L2 = 1.0000` and `cosine = nan` across all `39936/39936` weights and `32/32` biases.
   - All other blocks show normal fine-tuning drift from the parent (`mean rel L2` between `0.13` and `0.31`, high cosine similarity `0.946–0.999`).
3. **`c0_practice.md`**:
   - On the 300-line practice set, `CER = 1.0000`, `WER = 1.0000`, `collapsed lines = 300`, `length ratio = 0.000` (every line collapses to empty output because `C_3` kills the feature stream after `Cr3,13,32` + ReLU).

## Hypothesis & Edit Plan

- **H1 (block transplant of `C_3` from parent `kraken-german-xix-v1`):**
  - **Hypothesis:** `C_3.co.weight` and `C_3.co.bias` were zeroed out as a planted defect, severing the visual feature pipeline and causing 100% line collapse (`CER 1.0000`). Transplanting `C_3.co.weight` and `C_3.co.bias` from the parent model (`kraken-german-xix-v1`) will restore signal flow through the convolutional stack into the fine-tuned later layers.
  - **Edit:** `s.edit("transplant", ["C_3.co.weight", "C_3.co.bias"], hypothesis=h1, per_tensor={"C_3.co.weight": {"donor": P["C_3.co.weight"]}, "C_3.co.bias": {"donor": P["C_3.co.bias"]}})` followed by `s.keep_unmeasured(...)`.
  - **Edit class:** `transplant` (donor block restoration).
  - **Predicted effect:** Uncollapses all 300 lines; CER drops dramatically from `1.0000` to normal fine-tuned model levels (approximately `0.03–0.08`, a reduction of `~0.92–0.97` CER).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s09/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 94641 input / 1700 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
