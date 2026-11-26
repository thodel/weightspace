# Hypotheses — pack s03, task (b) (budget 0)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis from pack readings

1. **Practice-set failure pattern (`c0_practice.md`):**
   - Baseline CER is `0.0789` (WER `0.2135`, `1015` character edits across 300 lines), with **14 collapsed lines** and a low length ratio of **`0.941`**.
   - All top-10 confusions are pure deletions (`char → ∅`), led by `' ' → ∅` (177), `'e' → ∅` (81), `'n' → ∅` (53), `'i' → ∅` (38), `'r' → ∅` (37), `'s' → ∅` (28), `'t' → ∅` (26), `'.' → ∅` (26), `'o' → ∅` (23), `'u' → ∅` (21). This is the signature of an over-biased CTC blank class suppressing character emissions across the board.

2. **Weight inspection (`describe.md` and `diff_parent_to_subject.md`):**
   - In `describe.md`, the CTC blank row (`row 0` of `O_18.lin`) has weight L2 norm `4.637` and an abnormally large positive bias of **`+3.232`** (whereas non-blank rows have mean bias `-1.921 ± 2.010`, and the next highest bias in the entire layer is `'s'` at `+0.40`).
   - In `diff_parent_to_subject.md`, all convolutional, recurrent, and output weight tensors show normal fine-tuning drift from the parent `kraken-german-xix-v1` (output weight rows mean relative L2 change `0.1930`; blank row weight relative L2 change `0.2253`). However, the blank row bias shifted by **`bias Δ = +2.113`** relative to the parent (from `+1.119` in the parent to `+3.232` in the subject), consistent with a planted `+2.0` additive defect on `O_18.lin.bias[0]` on top of normal fine-tuning drift (`+0.113`).

## Hypothesis H1

- **Hypothesis:** `O_18.lin.bias[0]` (the CTC blank logit bias) carries a planted positive offset of `+2.0` (raising it to `+3.232`, `Δ = +2.113` over the parent's `+1.119`), which causes widespread character deletions (`length_ratio = 0.941`) and 14 collapsed lines.
- **Edit (`add` on `O_18.lin.bias`):** Subtract `2.0` from `O_18.lin.bias[0]` (adding a delta vector with `delta[0] = -2.0` and `0.0` elsewhere, restoring `O_18.lin.bias[0]` from `+3.232` to `+1.232`).
- **Predicted effect:** CER will decrease substantially by approximately `0.030` to `0.045` (from `0.0789` down to `~0.035–0.048`), eliminating the 14 collapsed lines, restoring the length ratio to `~1.00`, and removing the dominant `char → ∅` deletion errors.
