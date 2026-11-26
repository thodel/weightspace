# Pre-registration — 20261126-planted-k6-lstm16-x0.5-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k6-lstm16-x0.5-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s05/b/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — s05 / task (b)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`
- **Task:** (b), budget 0 (blind programmer)

## Diagnosis from pack readings

1. **Practice set performance (`c0_practice.md`):**
   - CER is `0.1993` (WER `0.4910`), with `24` collapsed lines and a low length ratio of `0.828`.
   - All top-10 confusions are character deletions (` ` → `∅`, `e` → `∅`, `n` → `∅`, `i` → `∅`, `r` → `∅`, `c` → `∅`, `s` → `∅`, `u` → `∅`, `t` → `∅`, `m` → `∅`), indicative of severe CTC under-emission / signal attenuation before the output layer.

2. **Tensor statistics (`describe.md`):**
   - In `L_12` and `L_14` (the first two BiLSTM layers), parameter norms and RMS values are completely normal and consistent with each other:
     - `weight_ih`: L2 ≈ `122.5–123.5` (for shape `[800, 400]`), RMS ≈ `0.217–0.222`
     - `weight_hh`: L2 ≈ `119.2–123.9` (for shape `[800, 200]`), RMS ≈ `0.298–0.310`
     - `bias_hh`: L2 ≈ `25.17–26.41` (for shape `[800]`), RMS ≈ `0.890–0.934`, abs max ≈ `2.50–2.59`
   - In `L_16` (the final BiLSTM layer, exact same shapes `[800, 400]`, `[800, 200]`, `[800]` as `L_14`), every non-zero tensor has an L2 norm and RMS that is almost **exactly half** (`0.5×`) of `L_14`:
     - `L_16.layer.weight_ih_l0`: L2 `68.42`, RMS `0.1210`
     - `L_16.layer.weight_hh_l0`: L2 `61.77`, RMS `0.1544`
     - `L_16.layer.bias_hh_l0`: L2 `11.98`, RMS `0.4236`, abs max `1.200`
     - `L_16.layer.weight_ih_l0_reverse`: L2 `64.03`, RMS `0.1132`
     - `L_16.layer.weight_hh_l0_reverse`: L2 `62.89`, RMS `0.1572`
     - `L_16.layer.bias_hh_l0_reverse`: L2 `12.38`, RMS `0.4378`, abs max `1.213`

3. **Comparison to parent (`diff_parent_to_subject.md`):**
   - For `L_12` and `L_14`, `bias_hh` hardly moved during fine-tuning (`rel L2` `0.032–0.042`, cosine `> 0.9991`), and `weight_ih` / `weight_hh` moved modestly (`rel L2` `0.21–0.30`, cosine `0.956–0.979`).
   - For `L_16`, `bias_hh` has the exact same high cosine similarity (`0.99897` and `0.99907`), and `weight_ih` / `weight_hh` have the exact same cosine similarities (`0.947–0.981`) as `L_12` and `L_14`! Yet every non-zero tensor in `L_16` has `rel L2` around `0.505–0.550`.
   - Mathematically, if a fine-tuned tensor $W_{\text{child}}$ has cosine similarity $c = \cos(W_{\text{parent}}, W_{\text{child}}) \approx 0.999$ and norm $\|W_{\text{child}}\| \approx \|W_{\text{parent}}\|$ before being scaled by $\alpha = 0.5$, then after scaling by $0.5$:
     $$\frac{\|0.5 W_{\text{child}} - W_{\text{parent}}\|}{\|W_{\text{parent}}\|} = \sqrt{1 - c + 0.25} \approx 0.505,$$
     and for $c \approx 0.956$ (`weight_ih`), $\sqrt{1.25 - 0.956} \approx 0.542$ — matching the table in `diff_parent_to_subject.md` (`0.5054`, `0.5059`, `0.5141`, `0.5201`, `0.5401`, `0.5499`) to three decimal places!
   - Thus, the planted defect is an exact `0.5×` scaling attenuation applied to the `L_16` BiLSTM block tensors (specifically `weight_ih_l0`, `weight_hh_l0`, `bias_hh_l0`, `weight_ih_l0_reverse`, `weight_hh_l0_reverse`, `bias_hh_l0_reverse`; `bias_ih` is zero). Halving the weights and biases inside the LSTM gates (`i, f, g, o`) dampens the pre-activations feeding into the sigmoids and `tanh`, shrinking the hidden state output $h_t$ of `L_16` that feeds into `O_18.lin`. Because `O_18.lin.bias` has a positive blank bias (`+1.232`) and negative character biases (`-1.921 ± 2.010`), attenuating $h_t$ causes the blank class to dominate, producing the massive deletion rate and 24 collapsed lines seen in C0.

---

## Hypothesis H1

- **Hypothesis:** All non-zero parameter tensors of the third BiLSTM layer `L_16` (`L_16.layer.weight_ih_l0`, `L_16.layer.weight_hh_l0`, `L_16.layer.bias_hh_l0`, `L_16.layer.weight_ih_l0_reverse`, `L_16.layer.weight_hh_l0_reverse`, `L_16.layer.bias_hh_l0_reverse`) were scaled down by a factor of `0.5`, attenuating the recurrent gate activations and output representation feeding `O_18`.
- **Edit (`scale`):** Scale the `L_16` parameter tensors by `factor=2.0` to exactly invert the `0.5×` attenuation and restore their true fine-tuned magnitudes.
- **Predicted effect:** Large reduction in CER on the practice set (from `0.1993` down by approximately `0.13–0.15` absolute to `~0.05–0.07`), elimination of the 24 collapsed lines, restoration of length ratio from `0.828` to `~1.00`, and resolution of the top character deletion confusions.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s05/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 116942 input / 3080 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
