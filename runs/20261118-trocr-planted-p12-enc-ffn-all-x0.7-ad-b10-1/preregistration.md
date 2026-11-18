# Pre-registration, 20261118-trocr-planted-p12-enc-ffn-all-x0.7-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s04/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s04/ad/program.py`, copied unchanged (sha256 `d80e79f08686474299568bc7b627b6b979143640a1c7aae2b679194c0109c4e4`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p12-enc-ffn-all-x0.7` = diffuse: every encoder FFN output projection (weights and biases) scaled (tensor digest `79402dcb01f311432e6f46a6c9df3684264140951e76e28ee0982e96a6286276`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration (task a, budget 10, with donor)

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools used: file read, shell (awk/grep on the pack only, py_compile). No model loaded, no pass run.

Reading: C0 CER 0.2736, 10 collapsed lines, length ratio 0.96, top confusions are deletions (space 401, e 71, ...). Diff to the donor: all 24 encoder `enc.NN.ffn.out.{weight,bias}` have norm ratio 0.70 (rel. distance 0.35, cosine ~1.0 for biases) while everything else is in the healthy range (embedding 0.527 is normal). Heads: none outlying.

H1 (main): every encoder FFN output projection (weight and bias, 12 layers) was scaled by 0.7. Edit class: scale, factor 1/0.7 = 1.4286 on all 24 tensors. Predicted: CER 0.274 -> about 0.20 to 0.23 (-4 to -7 points), collapsed lines down, space deletions down. If it fails (no CER gain beyond 0.5 pt): the shrink is not the cause of the deletions or the 0.7 is a healthy child artefact; I revert and try H2.
H2 (size refinement, only if H1 kept): extra scale 1.05 on the same tensors (net ~1.5). Predicted: -0.3 to +0.3 pt, kept only if the interval is below -0.5 pt. If it fails: factor 1/0.7 stays.
H3 (size refinement, only if H1 kept): extra scale 0.95 (net ~1.36). Predicted: within +-0.3 pt. Same rule.
H4 (only if H1 failed): partial restore, factor 1.2 on the same tensors. Predicted: -1 to -3 pt if the defect is real but the exact size differs. If it fails too: conclude the encoder FFN shrink is not what limits the practice CER; apply nothing.
H5 (only if H1 kept and CER still above 0.20): the remaining error is the child's own domain gap (the donor reads this hand worse than a healthy child); no further edit is justified by the readings, so I stop and keep unspent passes.

