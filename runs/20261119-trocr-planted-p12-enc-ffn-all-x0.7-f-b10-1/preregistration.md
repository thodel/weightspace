# Pre-registration, 20261119-trocr-planted-p12-enc-ffn-all-x0.7-f-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `f`: Claude Fable 5.2 (`claude-fable-5-2`), arm (a2), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s04/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s04/f/program.py`, copied unchanged (sha256 `239537818fd1e2a216c3a9be1bfa6baba8f5b6a6290cc267a03ab06bf3344089`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p12-enc-ffn-all-x0.7` = diffuse: every encoder FFN output projection (weights and biases) scaled (tensor digest `79402dcb01f311432e6f46a6c9df3684264140951e76e28ee0982e96a6286276`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model id: claude-fable-5-2. Settings: reasoning effort low, single session, task (a) budget 10 WITH the donor.
Tools used: Read and Bash (grep/awk/sed over the pack files and wsp/ops.py, wsp/session.py, wsp/metrics.py; py_compile). No model loaded, no pass run.

# Readings
- C0 practice: CER 0.2736, 10 collapsed lines, 401 deleted spaces, otherwise diffuse insert/delete of common letters: a global, non-token-specific degradation (weak visual evidence), not a token-row or single-head defect.
- Diff to parent: every `enc.NN.ffn.out.weight` and `enc.NN.ffn.out.bias` (NN = 00..11, 24 tensors) has norm ratio 0.700 with cosine 0.996 to 1.000 and relative distance 0.353 to 0.359 (healthy range 0.07 to 0.24). The biases have cosine 1.0000 and ratio 0.6998 to 0.7002: an exact multiplication by 0.7. All other encoder tensors have ratio 1.000 to 1.005; all decoder tensors are in the healthy range; tied embedding 0.527 (normal about 0.5), no token row with cosine below 0.5, no outlier head.

# Hypotheses
H1 (edit class: scale, rescaling of a tensor family). The output projections of the FFN in all 12 encoder layers were multiplied by 0.7. Edit: scale the 24 tensors `enc.00..11.ffn.out.{weight,bias}` by 1/0.7 = 1.428571. Prediction: CER falls from about 0.27 to 0.05 to 0.12 (a drop of at least 0.10, expected about 0.17), collapsed lines fall to 0 to 2, the space deletions fall by more than half; the keep rule is met. If it fails (rule refuses or CER drop below 0.03): the 0.7 ratio is not the cause of the errors, or the factor is wrong in size; go to H2.

H2 (edit class: scale; only if H1 is refused). Same tensors, per-tensor factor = parent norm / subject norm computed from the donor (about 1.42 to 1.43, exact per tensor). Prediction: same as H1 (CER drop of 0.10 or more). If it fails too: I conclude the defect is not repairable by rescaling this family and stop, leaving the model as it is (no transplant: the donor reads this hand worse).

H3 (edit class: scale, weights only; only if H1 and H2 are both refused). Scale only the 12 `enc.NN.ffn.out.weight` by 1/0.7, leave the biases. Prediction: CER drop of 0.05 or more. If it fails: stop, conclude nothing kept.

H0 (no further defect). After H1 is kept I expect no second defect: nothing else in the diff is outside the healthy range. I will make no further edit; unspent passes stay unspent. If CER after H1 is still above 0.15 I will note that a second, unseen defect is likely, but apply nothing without a reading that points to it.

Planned passes: 2 (C0, H1), at most 4.

