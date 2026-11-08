# Preregistration, pack bern-combined-v1 (programmer a3)

Model id: claude-sonnet-5-5. Settings: budget 10 passes (pass 1 = C0), keep rule = harness default (paired 95 % interval of CER(after) - CER(last kept) below 0). Written before any run; program.py is fixed in advance, branching only on keep/revert results.

Basis: C0 CER 0.1386; errors are dominated by deletions (reference -> nothing: space 195, `/` 98, e 87, n 78 ...), length ratio 0.949, blank row has the largest bias (+1.00) of all rows. The model is over-predicting CTC blank.

- H1 (additive, O_18.lin.bias[blank] -= 0.3): fewer deletions, CER falls by about 0.005-0.015 (absolute). If it fails (CER up): the deletions are not a blank-calibration problem but a feature problem; drop all bias edits on blank. If kept, try a further -0.3 (H1b, expected additional gain <= half of H1; fail means optimum is near -0.3).
- H2 (additive, bias of space row += 0.3): space is the top deletion (195 vs 99 insertions); CER falls 0.003-0.008. Fails -> space deletions arise from the features (wide gaps look blank-like), not output prior.
- H3 (additive, bias of `/` row += 0.3): `/` deleted 98 vs inserted 57; small gain 0.001-0.004 (or none: net deletion excess is only 41). Fails -> `/` errors are confusions with other glyphs, not calibration.
- H4 (interpolate, all non-output tensors toward kraken-bern-schlussredaktion-v1, alpha 0.25; donor rule mask unseen_by schlussredaktion): the sibling shares the parent and scores 0.118 vs 0.139 on all lines (0.182 on its unseen lines); linear mode connectivity of fine-tunes of one parent should give a CER fall of 0.005-0.02. Fails -> the siblings are not linearly connected at this distance (relative L2 about 0.2) or the sibling's advantage is only its own training lines.
- H5 (interpolate, output rows aligned by character, same donor and alpha, applied after H4 if kept): additional gain 0.001-0.005; fails -> output rows need the matching hidden features, no independent gain.
- H6 (if H4 kept: second step to alpha 0.5 total): extra gain 0.002-0.008, or revert (the optimum lies between 0.25 and 0.5).
- H7 (same for kraken-bern-hertwig-v2, alpha 0.25, only if H4 was refused; mask unseen_by hertwig-v2): the parent is worse (0.210), so I expect no gain (CER change -0.002 to +0.01); included as the only remaining use of budget.
- H8 (final, re-tuning blank bias by -0.2 after the merge if anything was merged): the merge shifts calibration; gain <= 0.003.
Expected total: CER 0.139 -> about 0.12-0.13 on practice. No hypothesis is expected to give more than 0.02.
