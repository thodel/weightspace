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
