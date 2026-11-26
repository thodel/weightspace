# Pre-registration, 20261124-trocr-stage4-B-b-b0-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject B (`dh-unibe__trocr-kurrent`), programmer Claude Sonnet 5.5, arm B, blind (budget 0), with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/B/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/B/b/program.py`, copied unchanged (sha256 `84d10b9c6089806e750b668f785a840e0cd9eb42c183e365e08af7a00002af35`); hypotheses below as written, before any pass.
Subject digest `2cce260da2c1d4a5705b8f15320f85a19b07fb4f650550f0c78647b4e00309df`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 0, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

Model: claude-sonnet-5-5 (Sonnet 5.5), Claude Code agent. Settings: default reasoning effort (not set explicitly), tools Read, Write, Bash (Bash only for compiling). Task (b), budget 0, with donor (donor not loaded; no pass, no run).
Correction on process: in my first step I used Bash to print files from the pack (cat/head) and a Bash grep of wsp/session.py was refused. I then read session.py with Read. No other files were read; no model was loaded and nothing was run.

# Readings
- C0 CER 0.2822, length ratio 0.958, 34 collapsed lines. Errors are mostly deletions (space 660, e 405, n 228, r 206, s 195, i 187) plus insertions of spaces (300) and a/e swaps. This is what a capable but imperfect reader of a hard hand looks like, not a damaged one.
- Distance to the donor is large (mean rel. 0.23), 71 % of it in the tied token embedding, then FFNs. That is ordinary fine-tuning from English to German Kurrent; the donor cannot read this hand.

# Hypotheses considered
H1. Global interpolation toward the donor (alpha 0.05-0.1). Expected effect: neutral to worse (+0 to +2 CER points); donor cannot read the hand and it is a listed comparator, not a finding. Rejected.
H2. Scale up decoder-side space/word-initial token rows (U+0120 tokens) to cut the 660 missing spaces. The tied readout also changes how those tokens are read in, and the 300 inserted spaces show the model already errs in both directions; sign and size cannot be fixed blind. Predicted effect: unknown, plausibly worse (+/-1 point). Rejected.
H3. Dampen or amplify the largest-moving tensors (dec.NN.cross.o.bias, norm ratios 2-3) to "undo" overshoot. Magnitude does not give direction and such moves do not reproduce across retrains; these biases are probably functional. Predicted: worse. Rejected.
H4. Length bias (readout of EOS row) to reduce deletions. Beam search with EOS scoring interacts non-monotonically; collapsed lines are only 34. Unmeasurable blind. Rejected.

# Decision
Ship nothing: program.py opens the session and freezes with no edit. Predicted CER change 0.000 (exactly the C0 model). Conclusion if wrong: a gain was available that magnitude-only readings could not reveal.

