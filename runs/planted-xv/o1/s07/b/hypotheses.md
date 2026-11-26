Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API). Reasoning effort: provider default.
Tools: read_file, list_dir, write_file (including py_compile), finish_round. No model execution or evaluation passes.

**H1 — the last BiLSTM was attenuated as a block.** In both directions, L16's recurrent weights have RMS 0.154–0.157 versus 0.299–0.310 at L14; its recurrent biases are 0.424–0.438 versus 0.931–0.934. Its input weights show a similar reduction. Weaker final character evidence fits the 0.828 length ratio, widespread deletions and 24 collapsed lines better than damage to a particular character row.

**Ship one scale edit:** amplify the six nonzero L16 tensors together, preserving the subject's features and gate balance. Use ×2 unless a nearby hundredth-step attenuation factor leaves a distinctive common float32 rounding signature in both directions, in which case invert that factor. Leave the blank and the rest of the head alone: deletion counts alone do not establish a blank-bias offset, and the unrelated donor cannot provide replacement features.

**Prediction:** fewer collapsed lines and deletions, with CER falling from 0.1993 by roughly 0.08–0.15 absolute. The exact gain and original scale cannot be inferred from RMS alone.
