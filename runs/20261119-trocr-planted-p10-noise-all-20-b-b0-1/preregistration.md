# Pre-registration, 20261119-trocr-planted-p10-noise-all-20-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s06/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s06/b/program.py`, copied unchanged (sha256 `8f272d3acd3b4640a8df761aa2e71501c49bb7c59361f8c3e19add22fefba1e8`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p10-noise-all-20` = diffuse: Gaussian noise on every encoder and decoder weight matrix, each at a fraction of its own norm (tensor digest `9b489cbf08b1caea4b2127425921a238a70273e568ac928500892f8a03fd64a7`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

model id: claude-sonnet-5-5; settings: default reasoning effort (low), tools: Read, Bash (grep/awk on my pack only, py_compile), Write. Task (b): budget 0, blind, with donor.

Readings: no non-finite values, no zero shares, per-layer norms smooth, no head with distance over 3x median, no zero/shrunk token rows, all tensors
within or marginally above the healthy step to the parent (dec.01-04 ffn.out 0.30 vs 0.24, smooth gradient across layers 0-11, norm ratio ~1.05, no single outlier;
tied embedding 0.53 = normal). C0 CER 0.2426 with 7 collapsed lines is plausible for a parent-derived model on a new hand; top confusions (space deletion, e/n/r) are generic.

H0: nothing is wrong. Edit: none. Predicted effect: none. Any repair would be a guess and could only harm.

