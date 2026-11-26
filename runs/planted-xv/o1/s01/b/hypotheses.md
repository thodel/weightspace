Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API).
Settings: provider-default reasoning; tools: read_file, list_dir, write_file (automatic py_compile), finish_round. Zero evaluation passes.

**H1 — erased second convolution.** Both `C_3` tensors are zero: changing the image cannot change the signal that reaches `C_6`. This explains all 300 empty readings without positing a blank-logit defect. Replace only the erased kernel and bias. Use Bern's second convolution as a stroke-feature prior, not as the missing weights: match its input channels by the signed similarity of first-layer filters, and estimate its output correspondence from the spatial weight patterns with which `C_6` consumes each channel. Keep all intact Kurrent weights.

**Prediction:** fewer than 300 empty lines and a partial CER reduction, roughly 5–25 percentage points from 100%. The diagnosis is much more certain than the magnitude: the unrelated model need not encode the same stroke features, even after channel matching. A return to normal reading accuracy is not predicted.
