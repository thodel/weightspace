Model id: claude-haiku-5-5

# Pre-registration: kraken-bern-combined-v1, programmer h1

Written from pack/ only (describe.md, diff_hertwig-v2_to_subject.md, diff_hertwig-v2_to_schlussredaktion-v1.md, c0_practice.md) and the API files, before any pass.

## Settings
- Subject: kraken-bern-combined-v1 (4.1 M parameters). Practice set: bern-val, 513 lines. Budget: 10 passes, pass 1 = C0 baseline (CER 0.1386 on all 513 lines, collapsed 27, length ratio 0.949).
- Keep rule: harness default, min_effect 0 (the 95 % paired interval of CER(after) - CER(last kept) must lie entirely below 0).
- Rule mask, donor-free edits and the baseline: lines unseen by kraken-bern-hertwig-v2. The subject was initialised from hertwig-v2, so hertwig-v2 is an ancestor of the subject even for edits that copy nothing from it. This is stricter than the brief requires.
- Rule mask, edits that copy from kraken-bern-schlussredaktion-v1: lines unseen by both hertwig-v2 and schlussredaktion-v1. If that set has fewer than 60 lines, the donor edits are skipped.
- Line membership comes from each item's `seen_by` at run time. The counts in c0_practice.md's header (Hertwig 212, Schlussredaktion 301) do not match its donor lines (hertwig trained on 212, schlussredaktion on 216), so I do not rely on the header counts.
- Output-row alignment: by character, with `wsp.align.codec_row_map` and `gather_rows`. Rows of characters the donor lacks keep the subject's own values. Row 0 (blank) is never taken from a donor.
- Each edit gets one measure and one keep/refuse decision. Refused edits are reverted by the harness. Planned passes: 1 baseline + 7 to 8 edits, within budget.

## Hypotheses
Magnitudes are absolute CER changes on the rule lines (the pack gives no CER for the mask), predicted by me, not measured.

- **H1 (output bias, blank row 0, add -0.25; edit class: output bias).** The 2408 character edits are mostly deletions (length ratio 0.949), and blank is already the highest-biased row (+1.00). Lowering it should turn marginal frames into characters. Predicted: CER falls by 0.005 to 0.020, deletions (∅ as hypothesis) fall by 20 to 40 %, length ratio moves toward 1. Conclusion if refused: blank bias is not the limiting term, so the deletions come from the conv/LSTM stack. Then only a smaller step (-0.10) is tried.
- **H1b (output bias, blank, second step: -0.25 more if H1 was kept, else -0.10; edit class: output bias).** Predicted: a further fall of 0.000 to 0.008, with a chance of insertions growing. The CER curve in blank bias should be U-shaped, with the minimum near -0.25 to -0.50 in total. Conclusion if refused: the optimum is at H1's step (or at -0.10 if H1 was refused).
- **H2 (output bias, space row +0.15; edit class: output bias).** Reference spaces missing in the hypothesis (195) outnumber spurious spaces (99). Predicted: CER falls by 0.001 to 0.004. Refusal is likely on power grounds alone. Conclusion if refused: spacing errors are not a bias-level problem.
- **H3 (output bias, slash row +0.15; edit class: output bias).** The same pattern as H2 (98 missing vs 57 spurious). Predicted: CER falls by 0.000 to 0.002. Refusal is the likely outcome. Conclusion: slashes are not a bias problem.
- **H4 (donor, output layer O_18 interpolated toward schlussredaktion-v1 with alpha 0.5; edit class: interpolate with donor; rule mask unseen by both).** The sibling has a better output calibration on its own unseen lines (CER 0.1820 on 297 lines). Predicted: CER falls by 0.000 to 0.010, kept with about 50 % probability. Conclusion if refused: the sibling's output layer does not transfer; the subject keeps its own output layer.
- **H5 (donor, all C_ and L_ tensors interpolated toward schlussredaktion-v1 with alpha 0.25; edit class: interpolate with donor; rule mask unseen by both).** Predicted: CER change of -0.005 to +0.002 (the body is further from the donor than the output layer), kept with about 40 % probability. Conclusion if refused: body interpolation is not safe at this alpha.
- **H6 (donor, same body interpolation at alpha 0.10; run only if H5 is refused; edit class: interpolate with donor).** Predicted: CER change within ±0.002; kept with about 30 % probability. Conclusion: no donor-based body edit transfers to this practice set.
- **Overall.** Predicted: the final artefact has a CER 0.005 to 0.020 below C0 on held-out lines with about 60 % probability, with H1 as the main contributor. There is about a 30 % chance that no edit is kept, in which case the final artefact equals C0.
