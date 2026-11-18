# Pre-registration, 20261118-trocr-planted-p08-shift-embnorm-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s02/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s02/ad/program.py`, copied unchanged (sha256 `2ba73df502eb9e2d9a53aac8cc184189223a47af9a2a8a838f3b6ff02399842d`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p08-shift-embnorm` = bias shift: a fixed random direction added to the decoder embedding layer-norm bias, a fraction of the norm of its weight (tensor digest `783d096bc6a72d33921dd7f21311e01168673f467232ee708d29df0764ab7cbc`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistered hypotheses, session s02, task (a), budget 10, with donor

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: default reasoning effort (low); tools: Read, Bash (pack files, brief, parse of the diff table only), Write. No model loaded, program not run.

Reading: C0 CER 0.2984, 33 collapsed lines, dominant errors are deletions (space 492, e 140, n 84 ...), length ratio 0.915. In the diff against the parent only one tensor is far outside the healthy range: `dec.embed.norm.bias` (rel. distance 1.71, norm ratio 6.31, cosine 0.12; norm 29.8 vs about 4.7 in the parent; its weight is untouched, 0.004). Everything else is in the healthy band (matrices 0.07-0.24; tied embedding 0.53, no zero or collapsed rows; no outlier heads; no non-finite values).

H1 (transplant): the decoder embedding LayerNorm bias was shifted or noised (probably an added offset to the bias). Edit class: transplant of `dec.embed.norm.bias` from the donor. Prediction: CER falls by at least 0.05 absolute (0.298 -> about 0.20 or lower), collapsed lines fall from 33 to under 15, deletion confusions shrink. If it fails (CER change within +-0.005): the bias is not the cause of the deletions; look at H4/H5.

H2 (centered correction, only if H1 is refused or clearly worse than H2 would be): the defect is a mostly constant offset c added to the bias. Edit class: add of -(mean of (subject - donor)) to every entry, keeping the donor-independent part of the fine-tune. Prediction: about the same gain as H1 within 0.01. Conclusion if it fails: the perturbation is not a constant shift; keep H1 or nothing.

H3 (partial interpolation, after H1 kept): a healthy child moves a LN bias by up to 0.13 relative; a full transplant puts the donor's bias in, which may be slightly worse for this hand than the child's own. Edit class: interpolate bias toward the child's own corrected state is not testable (no other reference); instead test alpha=0.75 toward the donor from the H1 state is not possible either. So H3 is only: scale the bias by 0.9 relative to the kept state. Prediction: change under 0.005 (no keep). Dropped unless budget remains; run last. Failing = expected, conclude H1 is already at optimum.

H4 (tied embedding left behind by the same defect): a noise-like perturbation of the embedding is unlikely (distance 0.53 equals the healthy value, no outlier rows), so prediction for interpolate(tokens, donor, 0.15): CER change within +-0.005, not kept. Conclusion if it unexpectedly helps by more than 0.005: the embedding was also damaged; try 0.3 once.

H5 (residual defects elsewhere): after H1, if CER is still above 0.20 and collapsed lines remain over 10, the donor reads this hand worse than a healthy child anyway, so I do not attempt other edits as nothing else in the diff is anomalous. Conclusion: stop and freeze.

Null case: if H1 is refused (CER not down by the margin), the program stops with the model unchanged apart from nothing kept.

