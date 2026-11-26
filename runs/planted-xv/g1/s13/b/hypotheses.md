Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer, provider default reasoning settings)
Tools used: `read_file`, `write_file`, `finish_round`

# Hypotheses — s13 task (b) (budget 0)

## Hypothesis 1 (`H1`): Sign-flip repair of row `'n'` in `O_18.lin.weight`
- **Diagnosis:** In `diff_parent_to_subject.md`, the output projection row for `'n'` in `O_18.lin.weight` has cosine similarity `-0.991` against the parent row and relative L2 change `2.042`, while its norm (`7.96 -> 8.33`) and bias change (`-0.142`) match normal child fine-tuning (`sqrt(8.33^2 + 7.96^2 + 2*8.33*7.96*0.991) / 7.96 = 2.042`). This shows the child's trained weight row for `'n'` was negated (`w_n -> -w_n`). Because `'n'` normalmente has negative dot product with features at non-`'n'` positions and positive dot product at `'n'` positions, negating the row causes massive spurious `'n'` insertions (`∅ -> n`: 7,667 in `c0_practice.md`, length ratio 1.611, CER 0.7456) and misses true `'n'` (`n -> m`, `n -> ∅`, `n -> r`).
- **Edit:** `set_rows` on `O_18.lin.weight` at index `[cm["n"]]` with `values = -T["O_18.lin.weight"][[cm["n"]]]` (restoring the exact sign of the child's `'n'` output weight row).
- **Edit class:** `set_rows` (output-layer row sign restoration).
- **Predicted effect:** Eliminates the ~7,667 spurious `'n'` insertions and restores true `'n'` recognition, reducing practice CER from `0.7456` by approximately `0.65–0.70` (down to healthy child CER `~0.05–0.09`).
