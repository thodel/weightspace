Model: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API via the operator's tool layer).
Settings: provider-default reasoning effort; tools: read_file, write_file (automatic py_compile), finish_round. Zero measurement passes.

**Hypothesis: the output affine layer has a common gain of 2.** For `á`, `Ä`, `š` and `é`, the child rows still point in the parent directions (cosines 0.998), yet each norm is 2.01–2.02 times its parent norm. Their negative bias shifts are consistent with amplification of negative biases as well. This looks like a change in logit temperature, not four class-specific failures: C0 has 6.50% CER and its leading confusions are deletions, not substitutions involving those rows.

**Ship:** multiply both `O_18.lin.weight` and `O_18.lin.bias` by 0.5, including blank and new characters. This preserves the child's learned directions and relative decision boundaries. Predict softer frame probabilities but approximately 0 percentage points change in CER: positive joint scaling leaves the frame argmax and hence the greedy CTC path unchanged. Do not transplant parent rows or adjust the space or blank bias; the deletion counts alone don't identify a faulty bias.
