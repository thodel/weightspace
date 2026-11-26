Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API).
Settings: provider-default reasoning effort; read_file, write_file (automatic py_compile), finish_round; no model execution or evaluation passes.

**Hypothesis: the last BiLSTM was scaled by 0.5 after training.** Its forward and reverse `bias_hh` vectors both have cosine about 0.999 to the parent but relative L2 displacement about 0.506. The child's RMS values (0.424 and 0.438) are about half those in the preceding layers. Both input and recurrent matrices show the same norm deficit in both directions while retaining high parent cosine. Weak last-layer character evidence is consistent with the short transcripts (length ratio 0.828) and widespread deletions, without requiring a fault in individual character rows.

**Ship:** multiply both directions' `weight_ih`, `weight_hh` and `bias_hh` in L_16 by 2 in one scale edit. Leave the zero input biases alone. This restores gate preactivation scales while retaining learned child-specific weights. **Prediction:** fewer deletions and collapsed lines, with CER falling by roughly 10–16 percentage points from 19.93% (blind estimate).

No further edits: the blank bias differs from its parent by only +0.113. The isolated relative change in the small C_3 bias does not identify a reversible fault.
