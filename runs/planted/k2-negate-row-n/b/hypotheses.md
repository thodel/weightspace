model: claude-sonnet-5-5; reasoning effort low; tools: Read, Bash (cat, py_compile) on the pack, brief and wsp/ops.py, wsp/session.py only. No passes.

H1 (fairly sure): output row `n` of O_18.lin.weight was sign-flipped. Evidence: diff shows `n` moved most, cosine -0.991 with the parent
(rel L2 2.04, i.e. ~|-x - x|/|x|), norm 7.96 -> 8.33 (a flip with ordinary fine-tuning drift), while every other row has cos > 0.88.
C0 reads: 7667 insertions of `n` (the flipped row fires where the blank/other classes should), n->m/r, e->n, i->n confusions, 79 collapsed lines,
CER 0.75, length ratio 1.61. Bias of `n` changed only -0.142, so the bias was not flipped.
Edit: O_18.lin.weight row cm['n'] multiplied by -1 (set_rows with the negated current row). Predicted: CER drops from 0.75 to roughly the healthy child level (~0.1-0.3), collapses vanish.

Not applied: the other large movers (« ¬ _ » …, cos ~0.9, bias -2.4) look like ordinary rare-character drift; C_3.co.bias rel change 1.09 is large
but it is a small tensor with cosine 0.68 and no sign pattern; I am not sure enough to touch it.
