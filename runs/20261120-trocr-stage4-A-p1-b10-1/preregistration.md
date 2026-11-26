# Pre-registration, 20261120-trocr-stage4-A-p1-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject A `from-kurrent-v2`, programmer Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm P1, with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/A/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/A/p1/program.py`, copied unchanged (sha256 `6c99a5a6b847d2fba22cc285452d479ecfd307dabc3b44a3bf876d598d820b49`); hypotheses below as written, before any pass.
Subject digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: default reasoning effort (not set by me), tools used: Read, Write, Bash (py_compile only). Task (a), budget 10, WITH the donor. Pack read: diff_magnitudes.md, c0_practice.md (describe.md not needed).

Readings: C0 CER 0.2307, length ratio 0.952, 29 collapsed lines. 1540 of 11809 edits are a deleted space, then deleted e/r/n/s/i: the model under-writes. The donor reads this hand about as well, so most of the gain I can hope for is small. 86 % of the movement from the donor is the tied embedding (rel. distance 0.53; deciles 4-9 sit at an almost uniform 0.60-0.63 with norm ratio above 1, which looks like a diffuse shift of rows, not training).
Keep rule: page-clustered 95 % interval entirely below -0.005. Most hypotheses below will fail it; that is the expected outcome.

H1. Interpolate all non-embedding tensors 30 % toward the donor (weight averaging of two fine-tunes of one hand, a mild soup). Class: interpolate, global (same as the comparator "single global interpolation toward the donor", so even a win is not a finding by itself). Predicted: CER -0.2 to -0.5 pts, more likely about 0. If it fails: the donor and the model are not in a common basin that helps, drop all donor-pull hypotheses (H5, H6).
H2. Scale the rows of all word-initial tokens (leading U+0120) of the tied embedding by 1.04. The decoder input goes through layernorm_embedding, so the input side is nearly scale invariant; the readout logit of those tokens grows, so a space is written more often. Class: scale on selected rows (readout bias). Predicted: fewer deleted spaces (1540 of 11809 edits), CER -0.3 to -1.0 pts, risk of inserted spaces. If it fails with CER up and spaces inserted: the space deletion is not a readout calibration issue; if it is flat, the effect is too small, then 1.08 is tried once.
H3. Scale the end-of-sequence row by 0.9 (fewer early stops; length ratio 0.952, 29 collapsed lines). Class: scale on one row. Predicted: length ratio toward 1, CER -0.1 to -0.4 pts; if it fails: early stops are not the reason lines are short, the deletions are inside lines.
H4. Replace the embedding rows in deciles 4-9 by movement (the diffuse shift, rows unlikely used for German) by the donor's rows. Class: set_rows from donor. Predicted: about 0 (rows hardly used), at best a few spurious insertions fewer; CER change within +-0.1. If it fails (CER up): those rows are in fact used and the donor's differ in a way that matters; no further embedding transplant.
H5. If H1 failed or was flat: interpolate only dec.ffn tensors 50 % toward the donor (largest non-embedding share of the move, 5.4 %). Predicted: about 0 to -0.3 pts. If it fails: the decoder FFN move carries no recoverable gain.
H6. Interpolate enc.ffn and enc.attn tensors 50 % toward the donor (encoder is where the image is read; move is 2.9 % + 1.2 %). Predicted: about 0 to -0.3 pts. If it fails: the encoder is already at least as good as the donor's.
H7. If H1 was kept: push it further, to 60 % total toward the donor. Predicted: a further -0.2 pts, possibly a reversal. If it fails: 30 % is the optimum on this set.
H8. If H2 was kept: scale the same rows by 1.08 in total. Predicted: further gain until insertions of spaces appear. If it fails: 1.04 is the optimum.
Conclusion rule: a kept edit that only reproduces the global interpolation comparator is not a finding; an empty program (everything reverted) is a legitimate result. At most 10 measure calls including C0.

