Model: claude-sonnet-5-5. Reasoning effort: low. Tools: Read, Bash (cat, ls, grep, py_compile only); no passes, no model loaded.

## H1 (applied): the top BiLSTM (L_16, both directions) has all its weights and biases scaled by about 0.5
Evidence: L_16 rms is about half of L_12/L_14 for every tensor (weight_ih 0.121 vs 0.22, weight_hh 0.154 vs 0.30,
bias_hh 0.42 vs 0.93). Against the parent, L_16 changes by rel L2 about 0.5 with cosine 0.999 on the biases (a pure rescale), while
L_12/L_14 move only 0.03-0.3. Practice reading: CER 0.199, length ratio 0.83, and the top confusions are all deletions
(character -> nothing), i.e. weakened emission confidence so the blank wins; fits an under-driven last recurrent layer.
Edit: scale all 8 L_16 tensors by 2.0 (bias_ih are zero, harmless). Predicted: CER falls substantially (0.20 -> roughly 0.10-0.14, toward the parent-child healthy level),
length ratio toward 1, collapsed lines fall from 24.
If wrong (a genuine fine-tune that just has smaller weights), CER would get worse; weak risk since the biases are exactly cosine 0.999.

## Not applied
- C_3.co.bias (rel 1.09, cos 0.68): large change but likely ordinary fine-tuning drift; no evidence of a defect.
- Output rows/biases: blank row and biases look normal; nothing to repair.
