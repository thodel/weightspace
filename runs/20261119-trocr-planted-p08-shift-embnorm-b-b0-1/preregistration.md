# Pre-registration, 20261119-trocr-planted-p08-shift-embnorm-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s02/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s02/b/program.py`, copied unchanged (sha256 `8091a3109acb190bb0998a70d5f13507e7c5a771c192d7abb996997911803741`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p08-shift-embnorm` = bias shift: a fixed random direction added to the decoder embedding layer-norm bias, a fraction of the norm of its weight (tensor digest `783d096bc6a72d33921dd7f21311e01168673f467232ee708d29df0764ab7cbc`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

model id: claude-sonnet-5-5; settings: reasoning effort low, tools used: Read, Bash (grep/awk on pack files, py_compile only), Write. No passes (budget 0), donor available.

Readings: every weight matrix and norm vector is within the healthy-child range (0.08-0.23 / <0.13), no zero rows, no non-finite values, no outlying heads, no collapsed token rows. The single outlier is `dec.embed.norm.bias`: rel. distance 1.71, norm ratio 6.31, cosine 0.12 to the parent (healthy: <=0.13). C0 CER 0.298 with many deletions fits a decoder input offset.

H1: a large shift was added to `dec.embed.norm.bias`. Edit: transplant the parent's bias (the only edit). Predicted: CER clearly down (several points to >10 points, deletions of space/e/n/r fall). Residual: the parent's value differs from the healthy child's by ~0.1, minor.

Not touched: tied embedding (cosine 0.86, rel 0.53, as in a healthy child), everything else.

