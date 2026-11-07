# Pre-registration — 20261107-val-bern-combined-v1-a3-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/bern-combined-v1/a3/program.py` copied unchanged.

1. **Subject:** `kraken-bern-combined-v1.mlmodel` — tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`. Donors: `kraken-bern-hertwig-v2` (trained on all 6 Hertwig val pages: 212 lines seen) and `kraken-bern-schlussredaktion-v1` (trained on 8 of the 11 Schlussredaktion val pages: 216 lines seen); donor rule applied through `rule_mask`.
2. **Practice set:** `data/practice/bern-val.jsonl`, 513 lines, sha256 `30d2b536deef178ab36d07e85180fa003f45fc015e657cda79b3e6b06031dbf2`. Source: the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1 (pages_val.lst; partition 0.9, page level), 6 Hertwig pages from the 22.10 export (correct images), 11 Schlussredaktion pages. **Not training data of the subject** (audit `data/practice/overlap-bern-val.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines, sha256 `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
