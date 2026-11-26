# Hypotheses — pack s06, task (b) (budget 0)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis from pack readings

1. **`describe.md`:** In `O_18.lin.weight` (297 × 400), the weakest row is `'e'` with L2 norm `0.00` (all other character rows have norms ≥ 7.68, with mean 14.787 ± 4.138; `<blank>` is 4.637).
2. **`diff_parent_to_subject.md`:** Comparing `kraken-german-xix-v1` (parent) to the subject on output rows aligned by character, row `'e'` has relative L2 change `1.000`, cosine similarity `0.000`, norm `7.64 → 0.00`, and bias Δ `-0.142`. All other shared output rows have cosine similarity ≥ 0.881 and healthy norms, and all conv/LSTM blocks show normal fine-tuning drift (cosines 0.945–0.999, except `C_3.co.bias` which has tiny norm 0.10).
3. **`c0_practice.md`:** Baseline C0 CER is `0.1623` (2,088 character edits across 12,867 reference characters). The top confusions are overwhelmingly dominated by missing or substituted `'e'`:
   - `e → ∅`: 910
   - `e → a`: 131
   - `e → i`: 90
   - `e → o`: 52
   - `e → r`: 36
   - `e → c`: 35
   - `e → u`: 32
   Total listed `'e'` confusions alone account for 1,286 of the 2,088 edits (~10.0 percentage points of CER). Because the weight row for `'e'` in `O_18.lin.weight` was zeroed out (and its bias potentially perturbed or left unable to fire without feature projections), the model cannot emit `'e'`.

## Hypothesis & Edit Plan

- **H1 (Output row transplant for `'e'` from parent):**
  - **Hypothesis:** The output projection row for character `'e'` in `O_18.lin.weight` was zeroed out (norm 7.64 → 0.00, cos 0.000) and its bias in `O_18.lin.bias` shifted by -0.142 relative to the parent model `kraken-german-xix-v1`. Because the recurrent backbone (`L_16`) remains closely aligned with the parent (`cos` > 0.946), transplanting the `'e'` row of `O_18.lin.weight` and the `'e'` entry of `O_18.lin.bias` from the parent model (`kraken-german-xix-v1`) into the subject's `'e'` row will restore the model's ability to recognize `'e'`.
  - **Edit:** `set_rows` on `["O_18.lin.weight", "O_18.lin.bias"]` at index `[cm["e"]]` using the donor values from `KrakenSubject(PARENT)` at index `[pm["e"]]`.
  - **Edit class:** Output-row donor transplant (`set_rows`).
  - **Predicted effect:** CER on the practice set will decrease by approximately **0.09 to 0.11** (from `0.1623` down to `~0.055–0.070`), eliminating the ~1,286 `'e'` deletion and substitution errors.
