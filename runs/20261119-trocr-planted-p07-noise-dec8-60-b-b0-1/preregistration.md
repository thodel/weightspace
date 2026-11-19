# Pre-registration, 20261119-trocr-planted-p07-noise-dec8-60-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s08/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s08/b/program.py`, copied unchanged (sha256 `22ff9124065bfdffc980c0d285b714f9236b20fbff3e8e7f2d1762a9a879804f`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p07-noise-dec8-60` = Gaussian noise on the weights of decoder layer 8 (self, cross, FFN), total norm a fraction of the layer's (tensor digest `1547142fbf5cd16953eec8cec5b0de50535bd5fd1d294fefe00309925161d6a1`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read, Bash (awk/grep/py_compile only), Write. Task (b): budget 0, with donor. No passes, program not run.

H1. Layer dec.08 weight matrices (self q/k/v/o, cross q/k/v/o, ffn in/out) carry added noise. Evidence: rel. distance to the donor 0.38-0.77
(healthy 0.07-0.24), norm ratio 1.07-1.35 (noise grows the norm), cosine 0.74-0.93, while biases and norms of the same layer and all other
layers look healthy. Edit: interpolate those 10 weight matrices 85% toward the donor (alpha=0.85), keeping 15% of the subject (the expected
healthy drift is about 0.15 rel., noise is much larger, so the donor is the better estimate). Predicted: CER 0.232 drops, probably by 3-8 points.
Biases and norm vectors left alone.

Not edited: embedding (0.53 distance, about normal), other layers, encoder (all normal), no zero or non-finite shares, no heads flagged.

