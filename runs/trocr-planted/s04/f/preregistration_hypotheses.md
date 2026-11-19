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
