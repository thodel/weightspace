# Pre-registration, 20261119-trocr-planted-p06-enc3-ffn-x0.5-b-b0-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `b`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s05/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s05/b/program.py`, copied unchanged (sha256 `146d688e9e27c5f26829fa9fbc41fe450b24f559c7d721aaac489f5b8d2a1aaa`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p06-enc3-ffn-x0.5` = encoder layer 3 FFN output projection (weight and bias) x0.5 (tensor digest `cfaca90a9d4f64012196994ac5d55049ccff0fff34eae08799c0158c6e74d6c2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 0, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `hypotheses.md`)

model id: claude-sonnet-5-5; settings: default/low reasoning effort; tools: Read, Bash (cat/awk/python on pack text only), Write. No passes, no model loaded.

H1. enc.03.ffn.out.{weight,bias} are scaled by ~0.5 (norm ratio 0.5027 / 0.5000 to donor, rel. distance 0.667, cosine 0.996 / 1.000; every other tensor is within the healthy range). Edit: scale x2.0 on both. Predicted: CER drops clearly (several points, from 0.2346); the dominant space-deletion confusion should shrink.

Not edited: tied embedding (distance 0.527, as healthy ~0.5, no zero or tiny rows), no heads flagged, no zeros or non-finite values, encoder layers 9-11 only moved a little (fine-tuning pattern, not a defect).

