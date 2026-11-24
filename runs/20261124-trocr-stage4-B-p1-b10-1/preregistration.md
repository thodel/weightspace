# Pre-registration, 20261124-trocr-stage4-B-p1-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject B (`dh-unibe__trocr-kurrent`), programmer Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm P1, with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/B/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/B/p1/program.py`, copied unchanged (sha256 `019f88dc6d40e6dca4136cdbab8f5a427681e1aabb277d1432bdc0297d778ff3`); hypotheses below as written, before any pass.
Subject digest `2cce260da2c1d4a5705b8f15320f85a19b07fb4f650550f0c78647b4e00309df`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: default reasoning effort (not overridden), tools used: Read, Write, Bash (py_compile only). Task (a), budget 10, with the donor.

Reading of the pack: the model is healthy. CER 0.2822; errors are dominated by deletions (space 660, e 405, n, r, s, i, c) and length ratio 0.958, 34 collapsed lines, so the main fault is under-generation (early EOS / skipped characters), not substitution. Prior: nothing to gain; every edit below is expected within noise or worse. Keep rule needs the clustered interval below -0.005.

H1. Edit class: scale (bias magnitude). The 12 `dec.NN.cross.o.bias` are 2.2-2.9x the donor norm, the largest norm growth in the model, and could be an overshoot that adds a constant offset to every decoder state. Scale by 0.9. Predicted: CER change -0.001 to +0.004 (most likely slightly worse). If CER rises, try 1.1 once (same bias, other side); if both fail, the bias magnitude is not a lever.
H2. Edit class: scale (single token row). Deletions with short outputs suggest too-early EOS or too-weak continuation. Scaling the EOS row (id 2, tied embedding) by 1.05 sharpens EOS logits both ways (mid-line more negative, at line end more positive) without touching other tokens. Predicted: change -0.002 to +0.002, length ratio +-0.003. If it fails, EOS is not the cause of the deletions.
H3. Edit class: scale (norm bias). `enc.norm_final.bias` has 2.6x donor norm and distance 1.05, the most moved encoder tensor. Scale by 0.9. Predicted: -0.001 to +0.005. If it fails, the encoder output offset is needed for the cross-attention.
H4. Edit class: interpolate (tied embedding toward donor). 71 % of the squared move is dec.embed.tokens; rows grew 17 %. A small step back (alpha 0.1) tests whether the embedding overshot. Predicted: +0.002 to +0.02 (worse, since a global interpolation toward an English model cannot read this hand better). If it fails as expected, this closes the global-shrink comparator in the embedding.

Conclusion rule: if no edit passes the keep rule, the result is "nothing to gain", program ends with the unchanged model. I do not read any direction into the magnitudes. Expected gain overall: none.

