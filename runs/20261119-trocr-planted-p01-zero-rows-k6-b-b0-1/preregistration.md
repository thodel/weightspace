# Pre-registration, 20261119-trocr-planted-p01-zero-rows-k6-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s03/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s03/b/program.py`, copied unchanged (sha256 `68c6cc51f13e48bcce30d1f85d80ea4d167f0873ed0a68d20bab2daee1afe0d1`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p01-zero-rows-k6` = the rows of the k most frequent token ids of the practice text set to 0 in the tied embedding (input and readout) (tensor digest `22b6a4c8bdabbc6fb8829ec7f60c743999ccb893f669d5a30b4724f9a488b0ca`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: low reasoning effort; tools used: Bash (read pack, grep/awk scan of diff table, py_compile), Read, Write. No model run.

H1: six rows of the tied embedding dec.embed.tokens are exactly zero (ids 6 ',', 4 '.', 90 't', 225 'en', 364 ' e', 2432 ' und'; cosine 0, norm 0). These tokens can be neither read nor written; matches C0 confusions (deleted '.', ',', 'e', space-before-word, 'n'). Edit: set those rows to the donor rows (the child's rows move ~0.5 relative elsewhere, so the donor row is the closest available estimate). Predicted: CER drops clearly (several points, perhaps 3-8).
No other anomaly: all other tensors within healthy distance, norm ratios 0.99-1.06, no head outliers, no non-finite. Nothing else applied.

