# Pre-registration, 20261120-trocr-stage4-A-b-b0-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject A `from-kurrent-v2`, programmer Claude Sonnet 5.5, arm B, blind (budget 0), with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/A/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/A/b/program.py`, copied unchanged (sha256 `6c571157a4840b11cef92d003453dceaac4ff0b5a9aad9e3c67bfa1c673483fd`); hypotheses below as written, before any pass.
Subject digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 0, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

Model id: claude-sonnet-5-5. Settings: default reasoning effort (not set explicitly); tools used: Read, Write, Bash (py_compile only). Task (b), budget 0, with donor.

# Hypotheses

The readings show healthy norms and a moderate donor move (embedding 0.53 relative distance, weight matrices 0.03 to 0.24, norm ratios near 1). The only
confusions are deletions (space, e, r, n, s) and e/i, a/e, n/r substitutions, which is ordinary handwriting error, not an edit-addressable fault.

H1. Interpolating toward the donor (globally or per component): predicted change about 0 (within +-0.3 CER points). The donor reads this hand about as well as the model, and a
global interpolation is itself a scored comparator, so a gain would not count as a finding. Not applied.

H2. Reverting the most-moved heads or the most-moved token rows to the donor: no evidence the move is harmful (magnitudes only, no direction), and the brief says such moves do not reproduce. Predicted effect: 0 or slightly worse. Not applied.

H3. Scaling any tensor (the norm ratios are all within 1.00 to 1.07; the largest are the cross.o biases at about 1.05 to 1.06): too small to matter, direction unknown. Not applied.

Conclusion: no edit is fairly sure to lower CER and a wrong edit would make the model worse without any check. Ship the empty program (freeze only). Expected CER change: 0.

