# Pre-registration, 20261119-trocr-planted-p12-enc-ffn-all-x0.7-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s04/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s04/b/program.py`, copied unchanged (sha256 `6592c1a59fa78f95a62adb8b82403cb829abb02e540ab002a4e7b7e2fb73ad85`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p12-enc-ffn-all-x0.7` = diffuse: every encoder FFN output projection (weights and biases) scaled (tensor digest `79402dcb01f311432e6f46a6c9df3684264140951e76e28ee0982e96a6286276`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: reasoning effort low (default), budget 0, blind, with donor. Tools used: Read, Bash (awk/grep/cat on own pack only), Write; no model loaded, program not run.

## H1 (shipped): all 12 encoder FFN output projections were scaled by about 0.7
Evidence: diff_parent_to_subject.md shows enc.00..enc.11 ffn.out.weight and ffn.out.bias, and only these, with norm ratio 0.700 to 0.704 (biases 0.6998 to 0.7002), cosine ~1.0 (biases exactly 1.0000), relative distance ~0.353 which is what a pure x0.7 gives. A healthy child moves matrices by 0.07 to 0.24 and keeps norm ratio near 1. CER 0.27 is high.
Edit: scale enc.NN.ffn.out.{weight,bias}, NN=00..11, by factor 1/0.7 = 1.428571.
Predicted effect: CER clearly down (several points, perhaps toward the 0.15 to 0.20 range).

## Not shipped
- dec.embed.tokens (rel. distance 0.53, cosine 0.86): matches the normal ~0.5 step of a healthy child; no outlier rows, no head outliers.
- decoder ffn.out weights 0.2 (norm ratio ~1.02) and dec.*.cross.o.bias norm ratio 1.06: within normal child range.
- No zeros, non-finite values or head-level anomalies found.

