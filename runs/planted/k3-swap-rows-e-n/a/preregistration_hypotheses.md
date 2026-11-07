# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5; reasoning effort low; tools: Read/Bash on the pack and wsp API only, no passes run.

Evidence: C0 CER 0.2443; top confusions are e->n (1364) and n->e (837), nothing else large (the next is 98). The output rows of `e` and `n`
have moved most from the parent (rel L2 1.19 and 1.10, cosine 0.36), every other row <= 0.63. Reads like the rows (and biases) of e and n exchanged.

H1 (edit class: set_rows, output rows swapped): swap rows e and n of O_18.lin.weight and entries of O_18.lin.bias. Predicted CER 0.244 -> about 0.08-0.12
(drop of ~0.13-0.16; e->n and n->e counts fall by >90 %). If it fails (CER not down, or e/n confusions remain): the swap is not the defect or only part of it, go to H2.
H2 (set_rows, weight only): swap the weight rows but leave the biases (if the bias swap is the harmful part, e.g. biases were not part of the corruption). Predict CER 0.09-0.13.
  If it fails: rows were not simply exchanged; go to H3.
H3 (interpolate / set_rows from donor, parent rows by character): replace e and n rows (weight+bias) with parent rows pm['e'], pm['n']. Predict CER 0.12-0.20 (the parent is worse on this hand, but the e/n
  confusion is removed); keep only if the rule accepts. If it fails: the defect is not in the output layer rows; stop, nothing further is applied.
H4 (kept state only, no edit): a collapse check: if collapsed lines rise after an edit, revert regardless.
Residual: 9 collapsed lines and ~98 missing spaces are baseline noise from the finetune, not targeted (space bias -7.85 is as in other rare rows).
