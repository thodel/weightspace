# Pre-registration — 20261107-val-bern-combined-v1-h1-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/bern-combined-v1/h1/program.py` copied unchanged.

1. **Subject:** `kraken-bern-combined-v1.mlmodel` — tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`. Donors: `kraken-bern-hertwig-v2` (trained on all 6 Hertwig val pages: 212 lines seen) and `kraken-bern-schlussredaktion-v1` (trained on 8 of the 11 Schlussredaktion val pages: 216 lines seen); donor rule applied through `rule_mask`.
2. **Practice set:** `data/practice/bern-val.jsonl`, 513 lines, sha256 `30d2b536deef178ab36d07e85180fa003f45fc015e657cda79b3e6b06031dbf2`. Source: the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1 (pages_val.lst; partition 0.9, page level), 6 Hertwig pages from the 22.10 export (correct images), 11 Schlussredaktion pages. **Not training data of the subject** (audit `data/practice/overlap-bern-val.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines, sha256 `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Haiku 5.5 (claude-haiku-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
