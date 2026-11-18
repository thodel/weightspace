# Pre-registration, 20261118-trocr-planted-p10-noise-all-20-an-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `an`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s06/pack_nodonor/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s06/an/program.py`, copied unchanged (sha256 `9bf49f1f61b88e5e9ddb2a34e288e50a6a20e3520c9c0b5e2b54eb0f0a108ad0`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p10-noise-all-20` = diffuse: Gaussian noise on every encoder and decoder weight matrix, each at a fraction of its own norm (tensor digest `9b489cbf08b1caea4b2127425921a238a70273e568ac928500892f8a03fd64a7`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: default reasoning effort (low), no donor. Tools used: Read, Bash (shell, text parsing of describe.md only), Write. No model loaded, no program run.

Reading: C0 CER 0.2426, dominated by deletions (space -> nothing 385, e -> nothing 62), length ratio 0.968, 7 collapsed lines. No zero shares, no non-finite values, no all-zero tensors. Norm anomalies against the layer medians: enc.00 attn.v (21 vs 37), enc.11 norm_attn.weight (6.9 vs 15.6), enc.00 norm_ffn.weight (44 vs 24), dec.11 norm_ffn weight/bias (67/3.5 vs 31/0.6). Many others (enc.09-11, dec.10-11 k bias) look like natural depth trends. Evidence is weak; the model may be healthy or lightly damaged. Each edit is kept only by the harness rule.

H1 scale enc.00.attn.v.weight x1.5 (norm 21 looks shrunk by ~0.6-0.7). Predicted CER -0.5 to -3 points if a scale defect; else +0.5 or worse. If it fails: enc.00 v is natural.
H2 scale enc.11.norm_attn.weight x1.6 (norm 0.44 of median). Predicted -0.5 to -3 points if defect; failure: last-layer norm is natural.
H3 scale dec.11.norm_ffn.weight x0.7 (norm 2.2x median). Predicted -0.5 to -2 points; failure: large last-layer norm is natural.
H4 scale all dec.NN.cross.o.weight x1.15 (deletions suggest too weak a read of the image). Predicted -0.5 to -2 points; failure: no global attenuation of the cross path.
H5 scale all enc.NN.ffn.out.weight and bias x1.05 (mild global encoder gain). Predicted within +-0.5 points; probably undone. Conclusion on failure: no global encoder gain defect.
H6 scale enc.norm_final.weight x1.1. Predicted within +-0.5; failure: final encoder scale fine.
Branching: after each measure, abandon if m.collapsed rises by more than 3 lines. Overall expectation: at most 1 kept edit; if none is kept, report no detectable defect.

