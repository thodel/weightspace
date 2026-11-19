# Preregistration (task a, budget 10, with donor)

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools used: file read, shell (awk/grep on the pack only, py_compile). No model loaded, no pass run.

Reading: C0 CER 0.2736, 10 collapsed lines, length ratio 0.96, top confusions are deletions (space 401, e 71, ...). Diff to the donor: all 24 encoder `enc.NN.ffn.out.{weight,bias}` have norm ratio 0.70 (rel. distance 0.35, cosine ~1.0 for biases) while everything else is in the healthy range (embedding 0.527 is normal). Heads: none outlying.

H1 (main): every encoder FFN output projection (weight and bias, 12 layers) was scaled by 0.7. Edit class: scale, factor 1/0.7 = 1.4286 on all 24 tensors. Predicted: CER 0.274 -> about 0.20 to 0.23 (-4 to -7 points), collapsed lines down, space deletions down. If it fails (no CER gain beyond 0.5 pt): the shrink is not the cause of the deletions or the 0.7 is a healthy child artefact; I revert and try H2.
H2 (size refinement, only if H1 kept): extra scale 1.05 on the same tensors (net ~1.5). Predicted: -0.3 to +0.3 pt, kept only if the interval is below -0.5 pt. If it fails: factor 1/0.7 stays.
H3 (size refinement, only if H1 kept): extra scale 0.95 (net ~1.36). Predicted: within +-0.3 pt. Same rule.
H4 (only if H1 failed): partial restore, factor 1.2 on the same tensors. Predicted: -1 to -3 pt if the defect is real but the exact size differs. If it fails too: conclude the encoder FFN shrink is not what limits the practice CER; apply nothing.
H5 (only if H1 kept and CER still above 0.20): the remaining error is the child's own domain gap (the donor reads this hand worse than a healthy child); no further edit is justified by the readings, so I stop and keep unspent passes.
