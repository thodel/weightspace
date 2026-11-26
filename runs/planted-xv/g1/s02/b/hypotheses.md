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
