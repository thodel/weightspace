# Pre-registration, 20261118-trocr-planted-p12-enc-ffn-all-x0.7-an-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `an`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s04/pack_nodonor/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s04/an/program.py`, copied unchanged (sha256 `56631b436aa1d3e19ea4c4b28ec37061fa30b8d71f99625c1db5ab676aac4d68`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p12-enc-ffn-all-x0.7` = diffuse: every encoder FFN output projection (weights and biases) scaled (tensor digest `79402dcb01f311432e6f46a6c9df3684264140951e76e28ee0982e96a6286276`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: reasoning effort low, tools used: Bash, Read, Write (only pack_nodonor/describe.md, c0_practice.md, the brief, wsp/ops.py and wsp/session.py signatures). No donor, MASK = s.unseen_by(). Budget 10, program uses at most 6 measures.

Readings: C0 CER 0.2736; dominant error is deleted spaces (401) and deletions of letters; no zeros, no non-finite tensors. Per-layer norms smooth except the outliers below.

H0 baseline C0. 
H1 (scale, dec.11.norm_ffn.weight+bias x0.5): the last-decoder-layer FFN LayerNorm has weight norm 67.5 (std 0.111) against ~31 (std ~0.02) in every other decoder layer, bias 3.5 vs ~0.6; looks like a scaled/perturbed LN feeding the readout. Predict CER -2 to -6 points if planted, ~0 or worse if natural. If it fails: try x0.75 once; if that fails too, conclude the gain is natural.
H2 (scale, enc.10.norm_ffn.weight+bias x0.6): encoder layer 10 FFN LN has norm 52.9/19.6 vs 20.5/15.0 in neighbours. Predict -1 to -3 points if planted; if it fails, conclude the late-encoder growth is natural (massive-activation channels), stop touching the encoder norms.
H3 (scale, enc.00.attn.v.weight x1.4): enc.00 attn.v norm 20.8 vs ~31 in layers 1-2 (shrunk?). Predict -0.5 to -1.5 points if shrunk; if it fails, concluded natural.
H4 (scale, enc.10.ffn.out.bias+weight x0.85) only if H2 kept: further relief of the same layer. Predict -0.5 point. Fails -> conclude H2 sufficed.
Null: if nothing is kept, the model is healthy or the defect is invisible in per-tensor norms (e.g. a few rows/heads); ship nothing.

