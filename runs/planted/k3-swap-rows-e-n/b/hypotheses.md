Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read and bash (reading the pack, brief and wsp API only; no model loaded, no passes).

# Hypotheses (blind, budget 0)

1. Output rows `e` and `n` of `O_18.lin.weight` / `O_18.lin.bias` are swapped. Evidence: top confusions e->n (1364) and n->e (837) dwarf all else; in the parent diff exactly `e` and `n` moved most (rel L2 1.19 and 1.10, cosine 0.36, vs <=0.63 for all others and ~0.2 mean); biases were not much changed (consistent with similar biases, so the swap hides in the weights). Edit: swap the two rows in weight and bias. Predicted: CER 0.244 -> roughly 0.08-0.12; the e/n confusions mostly vanish.
2. Nothing else is touched. The other differences from the parent (conv C_3 bias, new codec rows, general drift) look like ordinary fine-tuning; no evidence of further defects.
