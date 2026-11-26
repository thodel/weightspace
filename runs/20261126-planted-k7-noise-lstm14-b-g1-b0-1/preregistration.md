# Pre-registration — 20261126-planted-k7-noise-lstm14-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k7-noise-lstm14-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s02/b/program.py`, copied unchanged.

1. **Subject:** `k7-noise-lstm14.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 25 % of the block's; `runs/planted/plant.json`) — tensor digest `8caee20d9b623c22c56b655dafd5d95e8aa89e7703c7220073c79443973801bc`, file sha256 `4fe5cd0f681847910dbae3ee44dbe2ab2a2f538e6c93be363731499a65040bf8`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — s02 Task (b) (Budget 0, Blind Programmer)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** Provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis from Pack Readings

1. **Tensor Statistics (`describe.md`):**
   - All 34 tensors across convolutions (`C_0`, `C_3`, `C_6`, `C_9`), BiLSTMs (`L_12`, `L_14`, `L_16`), and the output projection (`O_18`) exhibit healthy, well-matched norms and RMS values (`C_*` weight RMS 0.184–0.201; `L_*` `weight_ih` RMS 0.221–0.242, `weight_hh` RMS 0.298–0.316, `bias_hh` RMS 0.847–0.934; `O_18.lin.weight` RMS 0.767, `O_18.lin.bias` RMS 2.777).
   - Output layer row norms (14.812 ± 4.069) and biases (-1.921 ± 2.010), as well as the CTC blank row (norm 4.637, bias +1.232), show no sign of scaling corruption, zeroing, or bias explosion.

2. **Parent-to-Subject Diff (`diff_parent_to_subject.md`):**
   - Every weight tensor aligns closely with the parent `kraken-german-xix-v1` (cosine similarity 0.919–0.9995 across all main weights and biases; `C_3.co.bias` has L2 0.10 and cosine 0.679, normal for a near-zero conv bias).
   - Across the 225 shared output rows, mean relative L2 change is 0.1930 (blank row relative L2 0.2253, bias Δ +0.113). The most-moved rows are rare symbols/punctuation with cosines 0.862–0.950. No row is inverted, zeroed, swapped, or anomalously shifted.

3. **Baseline Practice Readings (`c0_practice.md`):**
   - Practice CER is 0.0726 (WER 0.1923, length ratio 0.971, 934 edits over 12,867 reference characters).
   - Top-10 confusions are minor deletions/insertions proportional to natural character frequencies (` `, `e`, `n`, `s`, `i`, `r`, `.`, `t`, `o`), with no class collapse or spurious character flooding.

## Hypothesis, Edit, and Predicted Effect

- **Hypothesis (H0 — Healthy control / no defect present):** The subject model contains no planted weight defect; all layer statistics, parent-to-child diffs, and baseline practice readings are consistent with an uncorrupted fine-tuned model.
- **Edit:** None (`no-op`; 0 edits applied, freeze immediately).
- **Predicted Effect:** ΔCER = 0.0000 (practice CER remains 0.0726; avoids any unvalidated degradation on held-out evaluation).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s02/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 120469 input / 1524 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
