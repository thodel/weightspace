# Pre-registration, 20261119-trocr-planted-p05-zero-enc6-attn-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s01/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s01/b/program.py`, copied unchanged (sha256 `dc18a8eed1a89c24de3346ee9b8998c15e128a99f21433d520d1f47d185ebc02`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p05-zero-enc6-attn` = encoder layer 6 attention output projection (weight and bias) set to 0 (tensor digest `875ba809c9935636ca3433f3f7fece10d0701cbade4efae6a1500ddcc327bed2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

Model: claude-sonnet-5-5. Settings: low reasoning effort (default). Tools: Read and Bash (grep/awk/sort on pack files, py_compile only); no passes, program not run.

H1: `enc.06.attn.o.weight` and `enc.06.attn.o.bias` are all zero (norm ratio 0.0000, distance 2.0000 against the donor; all other tensors are within healthy range 0.003 to 0.23, no head or embedding-row outliers, no non-finite values). The practice CER of 0.33 with many pure deletions (space, e, n, r) fits a broken encoder layer.
Edit: transplant both tensors from the donor (a healthy child moves these ~0.1, so the donor is close to the original).
Predicted effect: large CER drop (several points to tens of points).
Not applied: dec.embed.tokens distance 0.527 is the normal ~0.5; the decoder FFN-out distances 0.2 to 0.23 are within the stated normal range.

