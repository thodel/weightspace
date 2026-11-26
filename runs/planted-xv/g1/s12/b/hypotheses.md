Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer, provider default reasoning settings)
Tools used: `read_file`, `write_file`, `finish_round`
Task: pack `s12`, task (b) (budget 0, blind programmer)

# Hypotheses

## H1 — Output-layer row swap between `e` and `n` (`O_18.lin.weight` and `O_18.lin.bias`)

- **Diagnosis from pack readings:**
  1. In `diff_parent_to_subject.md`, all convolutional (`C_0`–`C_9`) and recurrent (`L_12`–`L_16`) tensors show normal fine-tuning drift relative to the parent `kraken-german-xix-v1` (cosines 0.945–0.9999 across all major weights; `C_3.co.bias` has L2 0.10 so its relative L2 1.087 reflects small denominator norm).
  2. In the output layer `O_18.lin` (mean relative change 0.2019 across 225 shared rows), exactly two rows stand out as extreme directional outliers against the parent:
     - `e`: rel L2 = 1.189, cosine = 0.356, norm parent → child = 7.64 → 8.33, bias Δ = -0.039
     - `n`: rel L2 = 1.103, cosine = 0.361, norm parent → child = 7.96 → 7.57, bias Δ = +0.180
     Notice that the child's `e` norm (8.33) matches the parent's `n` norm (7.96) and the child's `n` norm (7.57) matches the parent's `e` norm (7.64), while their cosines with the opposite parent rows are symmetric (~0.36).
  3. In `c0_practice.md`, baseline C0 CER is 0.2443 (3,144 character edits on 12,867 reference characters), and the top two confusions are symmetric substitutions between `e` and `n`:
     - `e` → `n`: 1,364 occurrences
     - `n` → `e`: 837 occurrences
     Together these 2,201 confusions account for 70.0% of all character errors on the practice set.
- **Edit class:** `set_rows` (intra-subject row swap of the `e` and `n` rows in `O_18.lin.weight` and `O_18.lin.bias`).
- **Edit specification:** Swap row `cm["e"]` and row `cm["n"]` in `O_18.lin.weight` and `O_18.lin.bias` using `s.edit("set_rows", ["O_18.lin.weight", "O_18.lin.bias"], ...)`.
- **Predicted effect:** Eliminates the ~2,201 `e` ↔ `n` substitution errors on the 300-line practice set (12,867 reference characters), reducing practice CER by approximately **-0.171** (from 0.2443 down to ~0.073).
