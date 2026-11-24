# Pre-registration, 20261124-trocr-stage4-B-p2-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject B (`dh-unibe__trocr-kurrent`), programmer Claude Fable 5.2 (`claude-fable-5-2`), arm P2, with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/B/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/B/p2/program.py`, copied unchanged (sha256 `3435f6032cad29c05b242abfd12340dbaced6343224098454ac3e2a464ca6731`); hypotheses below as written, before any pass.
Subject digest `2cce260da2c1d4a5705b8f15320f85a19b07fb4f650550f0c78647b4e00309df`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

# Hypotheses: subject B, task (a), budget 10, with the donor (p2)

Model id: claude-fable-5-2 (Claude Code subagent). Settings: reasoning effort as set by the launching harness (not visible to me, not changed by me);
tools used: Read, Write, Bash (Bash only for one `py_compile` of `program.py`). Read: the brief, the three pack files, `wsp/ops.py`, `wsp/session.py`,
`wsp/run.py`, `wsp/trocr_subject.py`. Nothing was run, no model was loaded.

## What the readings say
- C0 on the practice lines: CER 0.2822, length ratio 0.958, 34 collapsed lines. The error is deletion-heavy: 660 lost spaces, and `e n r s i c` lost
  1,350 times together in the top ten; insertions appear only as 300 spaces. The model writes too little, and it does so on its own training pages, so it
  is under-fitted rather than over-fitted to them.
- `describe.md`: no zeros, no non-finite values, no layer out of line. `dec.11.norm_ffn` (norm 65 against about 31 elsewhere) is the last operation before
  the tied readout, so its gain is the temperature of the logits; that is by construction, not a defect.
- `diff_magnitudes.md`: 71 % of the squared move is in `dec.embed` (rows of deciles 3 to 9 at relative distance 0.84 to 1.18, norms up 18 to 25 %),
  the decoder FFNs next, the encoder little (0.09 to 0.26). Magnitudes only: I read no direction from them. The large relative moves of the `cross.o.bias`
  and `enc.norm_final.bias` vectors are of tensors with norm below 1 and are ignored.

Expectation overall: nothing is kept. The keep rule needs the whole page-clustered interval below -0.5 CER points, i.e. an effect of about one point,
and a healthy fine-tune rarely leaves that to a uniform edit. All predictions are on the practice set, against the last kept state.

## Hypotheses (order of the program; 10 passes at most, pass 1 is C0)

- **H1, cross-attention sharpness** (class: scale, `dec.NN.cross.q.{weight,bias}`, all 12 layers, factor 1.15 = attention logits x1.15).
  Lost characters come from a cross-attention that skips glyphs; sharper attention follows the line more strictly.
  Prediction: CER -0.1 to -0.4 points, deletions of `e n r s i` down, collapsed not up; not enough for the keep rule.
  Follow-up (one pass): if CER fell by more than 0.1 point, factor 1.32 in total; if it rose by more than 0.1 point, factor 0.87 (prediction then: -0.1 to -0.3).
  If it fails both ways: the sharpness of the alignment is at its optimum; the deletions are not an attention-temperature effect.

- **H2, end-of-sequence readout** (class: set_rows, the EOS row of the tied `dec.embed.tokens` x0.90).
  Length ratio 0.958 suggests lines end early; a shorter EOS row lowers its (positive, when chosen) logit. The row is also read in (it is the decoder
  start token in this family), which the embedding layer norm mostly absorbs.
  Prediction: length ratio up by about 0.005, CER -0.1 to -0.3 points; risk: more lines run on (collapsed up). Not enough for the keep rule.
  Follow-up: if CER fell by more than 0.1 point, x0.80 in total; if it rose, x1.10.
  If it fails: the missing characters are lost inside the line (the confusion table says so), not at its end.

- **H3, weight of the image evidence** (class: scale, `dec.NN.cross.o.{weight,bias}`, all layers, factor 1.15).
  The decoder adds the cross-attention output to a stream that also carries the language-model path; more image evidence against a prior that drops letters.
  Prediction: CER -0.1 to +0.3 points (direction uncertain, slightly more likely harmful because the following layer norm is tuned to the current gain).
  Follow-up: stronger (1.32) if it helped by more than 0.1 point, 0.87 if it hurt.
  If it fails: the balance of image and context is at its optimum.

- **H4, continue the fine-tune step in the encoder** (class: interpolate with the donor, alpha -0.15, i.e. extrapolation *away* from the donor, on
  `enc.NN.attn.{q,k,v,o}.weight` and `enc.NN.ffn.{in,out}.weight`, 72 tensors; a change of 1.5 to 4 % of each tensor).
  The model is under-fitted on its own training pages, so the step from the donor may be too short in the visual part. This is not the comparator
  (a global interpolation *toward* the donor) and uses this model's own step, not a direction read across models.
  Prediction: CER +0.0 to +0.5 points (harm more likely than gain; a gain above 0.3 point would surprise me). One pass, no follow-up.
  If it fails: the encoder's step has no slack along its own line; an under-fit is not cured by lengthening the step.

- **H5, logit temperature** (class: scale, `dec.11.norm_ffn.{weight,bias}` x1.25 = all logits x1.25; greedy choices unchanged, beam ranking and the
  stop decision change). Prediction: |change| below 0.2 point, sign unknown. Follow-up x0.80 if it rose, x1.56 if it fell by more than 0.1 point; only with
  the passes left. If it fails: the beam is insensitive to the temperature, as expected of a peaked model.

## Decisions
Every edit is decided by `s.keep` (the harness's rule), except that I revert myself when collapsed lines rise by more than 5 against the last kept state.
A kept edit that only matches what a random edit of the same size gives is not a finding; I expect to report "nothing kept".

