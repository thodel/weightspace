# Pre-registration, 20261124-trocr-stage4-B-p3f-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject B (`dh-unibe__trocr-kurrent`), programmer Claude Fable 5.2, arm P3, donor absent. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/B/pack_nodonor/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/B/p3f/program.py`, copied unchanged (sha256 `4be23bace04c65f6c170025a8aaa3728ae6f6e22f5f848ef6519a3b4a5047a92`); hypotheses below as written, before any pass.
Subject digest `2cce260da2c1d4a5705b8f15320f85a19b07fb4f650550f0c78647b4e00309df`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files absent.

## Hypotheses (`hypotheses.md`)

Model: claude-fable-5-2 (Fable 5.2), programmer subagent. Settings: reasoning effort as set by the caller (not overridden; default), tools used: Read, Write, Bash (py_compile only).
Task (a), budget 10, subject B, WITHOUT the donor (MASK = s.unseen_by(); no other model read, loaded or named).
Read: PROGRAMMER_BRIEF_B.md, B/pack_nodonor/describe.md, B/pack_nodonor/c0_practice.md, wsp/ops.py, wsp/session.py, wsp/trocr_subject.py.

# What the readings show

- describe.md: no tensor is degenerate (no zeros, no non-finite values; norms vary smoothly with depth). Nothing in the weights alone points
  at a defect. The only outlier, `dec.11.norm_ffn` (norm 65 against ~31), is the last operation before the tied readout and so sets the logit scale;
  it is expected to be large.
- c0_practice.md: CER 0.2822, length ratio 0.958, 34 collapsed lines. Errors are dominated by deletions (space 660, e 405, n 228, r 206, s 195,
  i 187, c 129) against 300 inserted spaces. The model writes too little: it skips characters and merges words.

Prior: the most likely outcome is no edit that passes the keep rule (95 % page-clustered interval entirely below -0.5 CER points). Every
edit below is a global, low-dimensional "knob" (one scalar); none is fitted to individual practice lines.

# Hypotheses (predicted change of practice CER; a keep needs about -1 point or better)

| # | hypothesis | edit class | edit | prediction | if it fails |
|---|---|---|---|---|---|
| H1 | Deletions come from the decoder leaning on its language prior instead of the image; a larger cross-attention write raises the image share. | scale, dec cross.o (all 12 layers, weight+bias) | x1.15 | -0.2 points (range -1.0 .. +0.5); not kept | the image/prior balance is at its optimum in this direction; the follow-up tests the other sign |
| H1b | Follow-up of H1 by its outcome: stronger (x1.15 again) if CER fell, opposite (x0.87) if it rose. | same | x1.15 or x0.87 | within +-0.5 points; not kept | the balance is flat or optimal; cross.o scale is not a lever |
| H2 | Same idea from the other side: a smaller self-attention write weakens the language prior. | scale, dec self.o (all layers, weight+bias) | x0.90 | 0 .. +0.3 points (slightly worse); not kept | the self-attention path is needed as is |
| H3 | Skipped characters come from diffuse cross-attention; sharper attention (larger query) tracks the line better and may cut loops (collapsed lines). | scale, dec cross.q (all layers, weight+bias) | x1.15 | -0.1 points (range -0.7 .. +0.5); collapsed lines +-5; not kept | attention sharpness is tuned; skipping is not an attention-temperature effect |
| H4 | Length ratio 0.958: the end token wins too early on some lines. Lower the end token's logit by 0.7 nat relative to the mean of the tokens of the hand, through the bias of the last layer norm (the readout has no bias; the tied row is left alone because the end token is also the decoder's start token). | add, dec.11.norm_ffn.bias (one direction) | -0.7 nat on EOS | -0.1 points (range -0.5 .. +0.3); length ratio up by <0.01; not kept | the missing characters are skipped inside lines, not cut at the end |
| H5 | Spaces are deleted 660 times and inserted 300 times: a net bias against word-initial tokens. Shift the logit of word-initial (U+0120) tokens against the others by +0.3 nat on average, same bias vector route. | add, dec.11.norm_ffn.bias (one direction) | +0.3 nat | -0.2 points (range -0.6 .. +0.4); fewer deleted, more inserted spaces; not kept | the space errors are visual (segmentation), not a readout bias |
| H6 | Beam search with a normalised score may prefer short, confident hypotheses; a sharper readout (pure logit temperature, weight and bias of the last norm scaled together) changes the ranking. | scale, dec.11.norm_ffn (weight+bias) | x1.25 | 0 +-0.2 points; not kept | the beam ranking is insensitive to temperature at this scale |
| F1, F2 | Two follow-ups on the two best not-kept knobs of H2..H6: double the step where CER fell, opposite sign where it rose. | as the parent | as the parent | at most -0.5 points; probably not kept | no global knob of this family yields a gain: report "nothing to gain" |

Decision rule in the program: an edit is offered to `s.keep` only if practice CER fell below the last kept state and collapsed lines did not rise by
more than 5; otherwise it is reverted. The harness rule then decides. Overall expectation: 0 kept edits (probability of at least one keep: about 10 %).
A kept edit of this kind would still have to beat random edits of the same size on unseen lines; a scalar knob chosen from the sign of the top
confusions is a weak finding at best.

