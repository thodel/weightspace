# s06, task (b): blind programmer (budget 0)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort (agent run under Claude Code); tools used: Read/Bash
(cat, ls, grep, wc) on the three pack files, the brief and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
`wsp/align.py`; `python -m py_compile` on the program. No pass, no model loaded, no other folder read.

## Reading of the pack

- `describe.md`: output row `e` has norm **0.00** (every other common letter sits at 7.8-8.3: `r` 7.80, `s` 8.05, `t` 8.08, `i` 8.17,
  `n` 8.33, `a` 8.33). `diff_parent_to_subject.md`: `e` moved most, rel L2 1.000, cosine 0.000, norm parent -> child 7.64 -> 0.00,
  bias Δ only -0.142. That is a zeroed weight row with the bias left (nearly) alone.
- `c0_practice.md`: CER 0.1623, 2088 edits; `e`->∅ 910, `e`->a 131, `e`->i 90, `e`->o 52, `e`->r 36, `e`->c 35, `e`->u 32: about
  1290 of the 2088 edits are missed `e`s. With a zero weight row the `e` logit is a constant (its bias), so the blank (bias +1.23)
  or a neighbouring vowel wins at every `e`. Length ratio 0.900 and 9 collapsed lines are consistent with ~7 % of the characters
  never being emitted.
- Everything else looks like ordinary fine-tuning drift: all three BiLSTM blocks moved uniformly (mean rel L2 0.137-0.141, cosines
  0.95-0.98), conv blocks 0.13-0.31. Two things I noticed but am **not** acting on: `C_3.co.bias` rel L2 1.09 / cosine 0.68, but the
  tensor is tiny in both models (child L2 0.10, implied parent ~0.07) so a large relative change there says little; and
  `L_16.layer.weight_ih_l0` has abs max 4.70 where its siblings top out at ~2, but its rel L2 (0.296) is in line with the other
  input matrices, so if anything is off it is one or a few weights I cannot locate from these readings. Neither shows up in the
  confusions, and a blind edit on either would be a guess.

## Hypotheses

| # | hypothesis | edit | predicted effect | applied? |
|---|---|---|---|---|
| H1 | The `O_18.lin.weight` row of `e` was zeroed; the model can never emit `e`. | `set_rows` on `O_18.lin.weight`, row `cm['e']`, with the parent's `e` row (`pm['e']`, aligned by character; norm 7.64, in line with the child's other common letters). Child bias kept (it is within 0.14 of the parent's). | CER down by roughly 0.06-0.09 (0.162 -> ~0.07-0.10): most of the 910 deletions and a good part of the ~380 `e`->vowel substitutions recovered. The parent row sits in a feature space that drifted ~0.3 rel L2, so it is an imperfect detector; some `e`s stay missed and a few `a`/`i`/`o` may flip to `e`, but a zero row is strictly worse than any reasonable `e` detector. Collapsed lines and length ratio should move towards 0 and 1.0. | yes |
| H2 | `C_3.co.bias` was tampered with. | none | not confident; both tensors are near zero, no confusion pattern points at the convs. | no |
| H3 | A spike in `L_16.layer.weight_ih_l0` (abs max 4.70). | none | not confident; cannot locate it from the pack and its rel L2 is normal. | no |

Guard inside the program: the edit is only applied if the child's `e` row really has norm < 1e-3 and the parent's `e` row has a
sane norm (3-15). If either check fails the program notes it and freezes without editing.
