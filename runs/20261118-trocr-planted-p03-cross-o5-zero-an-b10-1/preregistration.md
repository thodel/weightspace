# Pre-registration, 20261118-trocr-planted-p03-cross-o5-zero-an-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `an`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s07/pack_nodonor/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s07/an/program.py`, copied unchanged (sha256 `407f505222ec4d63711a0fff888e62f9f54b0a1d0022ffe8eb12650415a08369`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p03-cross-o5-zero` = decoder layer 5 cross-attention output projection (weight and bias) x0.5 (tensor digest `0cd9334534d039638da99733c2964c80bd6ba017010a899024ba8d23b668f438`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: reasoning effort low (default agent settings), budget 10 passes, no donor. Tools used: Read, Bash (reading the pack, grep/python table scan of describe.md, py_compile only), Write.

Reading: describe.md shows exactly one hard anomaly: `dec.05.cross.o.weight` and `dec.05.cross.o.bias` are all zeros (share of zeros 1.0; the other 23 cross.o tensors have norm ~33 and ~0.6). Layer 5 therefore contributes nothing from the encoder. C0 CER 0.232 with 416 deleted spaces and many deleted letters (hypothesis shorter than reference, length ratio 0.953) fits a decoder that has lost part of its image reading. Other outliers (enc.9-11 norms, dec.11 norm_ffn, some k biases) look like a natural growth of norms with depth / ordinary features, and are not hypotheses except H5.

H1 (edit class: set/interpolate from neighbours, `add` onto zeros): rebuild dec.05.cross.o.{weight,bias} as 0.5*(dec.04 + dec.06) cross.o. Neighbouring layers read in a similar basis (residual stream), but head subspaces do not align per layer, so only a partial repair. Predicted CER change -0.01 to -0.04 (e.g. 0.232 -> ~0.205); may be neutral. If it fails (not kept): the heads of layer 5 are layer-specific, neighbour copies inject noise; try single neighbours (H2), then give up on this tensor and leave it zero (a zero output projection is harmless compared with a wrong one).
H1b (scale): if H1 is kept, the 0.5 average may be too weak/strong: multiply the rebuilt tensor by 1.6 (predict -0.003 to -0.01 more). If not kept, revert, keep H1 level.
H2 (set from one neighbour): dec.05.cross.o := 0.6*dec.04 (then 0.6*dec.06) if H1 failed. Predict -0.005 to -0.02 for the better of the two; conclusion if both fail: zero layer is not repairable without a donor.
H3 (scale, only if a repair was kept and budget remains): per-head attenuation is not identifiable without the donor, so none is tried.
H5 (scale, weak): dec.11.norm_ffn.weight norm is 67 vs ~31 median and enc.10.norm_ffn.bias 19.6 vs 6.6: possible amplified/altered tensors. Test only dec.11.norm_ffn.weight x0.8. Predict 0 +- 0.005; low prior (<15 percent) because last-layer norms are commonly larger. If it does not clear -0.005 it is reverted and I conclude these are natural.
Null: if H1/H2 all fail, the final model is the unchanged weights apart from kept edits; no claim of a second defect.

