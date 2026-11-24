# Pre-registration, 20261120-trocr-stage4-A-r-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject A `from-kurrent-v2`, programmer arm R, replay, with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/A/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/A/r/program.py`, copied unchanged (sha256 `22cfb6a04154603064522600df73c9edcd004384e347ffe31d514aad5b88c49b`); hypotheses below as written, before any pass.
Subject digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: default reasoning effort (not set explicitly); tools used: Read, Write, Bash (py_compile only). Task (a), budget 10, with the donor. No pass spent in writing this.

Reading of the pack: C0 CER 0.2307, length ratio 0.952, 29 collapsed lines; the dominant error is a deleted space (1540 of 11809 edits), then deleted letters (e, r, n, s, i). The model writes too little. The donor move is 86 % in the tied embedding, with rows of deciles 3-9 at an almost constant relative distance of about 0.6 (looks like a uniform drift of rarely used rows, not a signal), and small elsewhere. I read no direction into any magnitude.

Budget plan: 1 pass C0, up to 6 edit passes, 3 spare.

H1 (pseudo output bias toward word-initial tokens). The tied embedding has no output bias, but dec.11.norm_ffn.bias (the last layer's output shift, no final norm after it) feeds only the readout, so adding a vector d there shifts every logit by e_t . d. Take d along (mean embedding of U+0120-initial tokens minus mean of all rows), scaled so that the mean logit shift of word-initial tokens is +0.5 nats. Edit class: add on one bias vector (readout bias synthesis). Prediction: CER down 0.5 to 1.5 points on practice (fewer deleted spaces; the 1540 deletions are 3 % of reference characters), if the deletions are a calibration issue of the readout. If it fails (CER rises or no change): the missing spaces are not a readout-prior problem (they are decided by the image evidence), and I stop pushing the space prior; a rise by insertion of spurious spaces would show in confusions as new space insertions. If kept, try +1.0 total; if reverted but the point estimate fell, try +0.25.

H2 (end-of-sequence suppression). Same trick with d = -e_eos scaled to lower the EOS logit by 0.7 nats. Edit class: add on one bias vector. Prediction: CER down 0.2 to 0.8 points through fewer early stops (length ratio 0.952, 29 collapsed lines). If it fails: early ends are not the source of the short outputs; the shortfall is inside lines (deleted letters), and I conclude the length deficit is evidence-driven.

H3 (partial move back toward the donor over all non-embedding tensors, alpha 0.25). Edit class: interpolate, global; this is the plain comparator the final judgement uses, so by itself it is not a finding; I run it to see whether any gain is above what H1/H2 give. Prediction: CER change between -0.3 and +0.3 points (two fine-tunes of about equal quality; weight averaging can give a small gain). Kept only if the rule is met. If kept, one more step of the same size (total about 0.44). If it fails: the donor and the model sit in different basins on these tensors or there is nothing between them.

H4 (shrink decoder FFN outputs toward the donor only). Not run unless H3 is kept; skipped otherwise.

What I will conclude overall: if nothing passes the half-point page-clustered rule, nothing is gained, and the program ends with an unchanged model (freeze after reverts). That is the expected outcome (most likely 70 %).

