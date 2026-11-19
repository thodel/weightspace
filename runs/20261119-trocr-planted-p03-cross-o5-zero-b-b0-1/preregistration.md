# Pre-registration, 20261119-trocr-planted-p03-cross-o5-zero-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s07/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s07/b/program.py`, copied unchanged (sha256 `7b5d7f7ce55aaa838fbb470e9b8ead4e5d6f7e41864a6007e1713f10488c5ba9`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p03-cross-o5-zero` = decoder layer 5 cross-attention output projection (weight and bias) x0.5 (tensor digest `0cd9334534d039638da99733c2964c80bd6ba017010a899024ba8d23b668f438`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: default/low reasoning effort, budget 0 (blind, with donor). Tools: Read, Bash (cat/awk/grep, py_compile only), Write.

H1 (apply): `dec.05.cross.o.weight` and `.bias` are all zero (norm ratio 0.0, rel. distance 2.0 to the donor), so decoder layer 5 cross-attention contributes nothing. Edit: transplant both tensors from the donor. Predicted: CER drops by several points from 0.232 (the confusions are dominated by dropped spaces and letters).
Not applied: every other tensor lies within the healthy-child range (matrix distances 0.08-0.24, norm ratios 1.0-1.06); no head outliers; tied embedding distance 0.53 is normal (~0.5), no zero or collapsed rows. No other defect is evident.

