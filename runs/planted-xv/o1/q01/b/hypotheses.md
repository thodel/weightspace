Model ID: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API via the operator's tool layer).
Settings: provider-default reasoning; tools used: read_file, write_file (automatic py_compile), finish_round. Budget: zero passes.

**Hypothesis: the adapter is being applied at twice its intended strength.** At rank 64, q01's alpha of 256 gives a multiplier of 4, versus 2 for both same-recipe siblings. The norm ratios cluster near 1.8 across attention and MLP layers, with no isolated module driving the excess. C0's 28 looping pages out of 32 are consistent with a gain error affecting the whole decoder.

**Edit:** Multiply every LoRA B by 0.5 and leave every A unchanged. This makes each update exactly what q01 would produce at alpha 128, preserving its learned directions and relative module strengths. The resulting total update norm (117.50 versus 127.35 and 135.65 for the siblings) does not need to match either independent run exactly; there is no clear evidence for a second, local correction.

**Prediction (unmeasured):** Practice CER should fall from 2.50 to below 0.50 (a drop of more than 2.0 absolute), with fewer than 10 pages reaching the cap instead of 29. The direction is more certain than the magnitude because individual looping pages can dominate CER.
