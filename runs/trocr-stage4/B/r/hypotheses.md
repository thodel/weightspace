# Hypotheses: TrOCR stage 4, subject B, arm R (budget 10, with the donor)

Programmer: Fable 5.2 (`claude-fable-5-2`), run as a Claude Code subagent. Reasoning effort: the harness default (not shown to me, not changed by me).
Tools used: Read (the brief, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/trocr_subject.py`, the three pack files), Write (this file and `program.py`),
Bash only for `ls` of my own folder, `wc -l` of `describe.md` and `python -m py_compile`. No model loaded, no pass run, no other folder read.

## What the pack says (and what it does not)

- C0: CER 0.2822, WER 0.568, 34 collapsed lines, length ratio 0.958. This is a weak reader, not a finished one in the sense of a low CER:
  9 of the top 10 confusions involve a deletion or an insertion, and deletions dominate (space 660, e 405, n 228, r 206, s 195, i 187, c 129 against 300 inserted
  spaces). The model writes too little: it skips glyphs and merges words. Only `a`/`e` is a real substitution pair, and it is nearly symmetric (170 / 138),
  so no token-row edit can win on it.
- Diff: 71 % of the squared move sits in `dec.embed` (the tied embedding: rows of deciles 3 to 9 moved by about 0.9 of their norm with norm ratio 1.2, which is
  what a common suppressive shift of unused tokens looks like; nothing to gain there). The decoder FFN grew (norm ratio 1.2 to 1.38), the encoder moved
  least (0.09 to 0.26, the last encoder layers least of all). All of this is magnitude; I read no direction into it.
- Nothing in `describe.md` is broken: no zeros, no non-finite values, no layer with an odd norm.

My prior: nothing passes the keep rule (interval entirely below -0.5 points). Each hypothesis below is a one-parameter, function-level edit that a random
edit of the same size cannot imitate and that a global interpolation *toward* the donor does not contain.

## Hypotheses (edit class; predicted change of practice CER; conclusion if it fails)

- **H1 (interpolate, alpha < 0: extrapolation away from the donor).** The fine-tune is under-converged (CER 28 %), so a small step further along
  donor -> subject, on every tensor except the two decoder embedding tables, continues the adaptation: `w + 0.10 (w - donor)`.
  Prediction: -0.2 to -0.6 points, more likely inside the noise; not kept by the rule. If CER rises: the fine-tune end point is not short of a
  minimum along its own line, and the direction of the move carries no usable "more of the same".
  - **H1b**, follow-up. If H1 was kept: a second step of 0.10 (prediction: another -0.2). Otherwise: encoder only, `w + 0.20 (w - donor)` on the `enc.*` tensors,
    because the encoder is the part that moved least while the script changed most (prediction: -0.1 to -0.4, probably noise). If it fails: the
    encoder is not under-adapted along its own move either; H1 is closed.
- **H2 (scale).** The decoder under-uses the image relative to its own context (it skips glyphs): raise the gain of every cross-attention output,
  `dec.NN.cross.o.{weight,bias}` x 1.15, all 12 layers. Prediction: deletions down, CER -0.2 to -0.5 points; could as well be +0.3.
  - **H2b**, dose: x 1.30 from the kept state if H2 lowered CER, x 0.87 if it raised it (prediction for the opposite direction: within +-0.3). If both
    directions lose: the cross-attention gain sits at its optimum; closed.
- **H3 (scale).** Cross-attention is too diffuse to step glyph by glyph (skipped minims e, n, r, i, c): sharpen it by a temperature,
  `dec.NN.cross.q.{weight,bias}` x 1.15, all 12 layers (the attention logits scale by 1.15, nothing else changes). Prediction: -0.1 to -0.4 points, or noise.
  - **H3b**, dose: x 1.30 if H3 lowered CER, x 0.87 if it raised it. If both lose: the temperature is at its optimum; closed.
- **H4 (set_rows on the tied embedding, one row).** Part of the missing text is an early end of line: lower the constant part of the end-token logit
  by about 1.5 nats by moving the `</s>` row against the direction of `dec.11.norm_ffn.bias` (the only input-independent part of the final decoder state;
  step capped at 0.35 of the row's norm). Prediction: length ratio up by under 0.01, CER -0.1 to -0.3 points; weak, because the bias (norm 3.6) is a small
  part of the final state (gain norm 65) and because `</s>` is also the decoder start token, whose input embedding moves with the row. If CER rises
  or collapsed lines rise: the deletions are inside the line, not at its end; closed.
  - **H4b** only if H4 lowered CER: the same step once more.

## Budget plan (10 passes, written in advance)

| pass | what |
|---:|---|
| 1 | C0 baseline |
| 2 | H1 extrapolation 0.10, all tensors but `dec.embed.tokens`, `dec.embed.pos` |
| 3 | H1b (second step if kept, else encoder only 0.20) |
| 4 | H2 cross.o x 1.15 |
| 5 | H2b dose (x 1.30 or x 0.87) |
| 6 | H3 cross.q x 1.15 |
| 7 | H3b dose (x 1.30 or x 0.87) |
| 8 | H4 `</s>` row shift |
| 9 | H4b only if H4 lowered CER; otherwise unspent |
| 10 | unspent (reserve; no edit is planned for it) |

Decisions: an edit whose practice CER is not below the last kept state's, or that adds more than 5 collapsed lines, is reverted by the program; otherwise
`s.keep` is called and the harness rule decides (refused = undone). A dose step is always applied to the last kept state, never on top of a refused edit.
If a pass fails, the pending edit is reverted and the session is frozen as it stands. Expected final state: the unchanged model.
