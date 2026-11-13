# Pre-registration — 20261107-val-bern-combined-v1-a2-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/bern-combined-v1/a2/program.py` copied unchanged.

1. **Subject:** `kraken-bern-combined-v1.mlmodel` — tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`. Donors: `kraken-bern-hertwig-v2` (trained on all 6 Hertwig val pages: 212 lines seen) and `kraken-bern-schlussredaktion-v1` (trained on 8 of the 11 Schlussredaktion val pages: 216 lines seen); donor rule applied through `rule_mask`.
2. **Practice set:** `data/practice/bern-val.jsonl`, 513 lines, sha256 `30d2b536deef178ab36d07e85180fa003f45fc015e657cda79b3e6b06031dbf2`. Source: the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1 (pages_val.lst; partition 0.9, page level), 6 Hertwig pages from the 22.10 export (correct images), 11 Schlussredaktion pages. **Not training data of the subject** (audit `data/practice/overlap-bern-val.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines, sha256 `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, bern-combined-v1, programmer a2 (claude-sonnet-5-5)

Settings: budget 10 passes (pass 1 = C0), keep rule = paired 95 % interval below 0, donor rule applied for the one donor edit (schlussredaktion-v1, mask unseen_by).
Basis (c0_practice.md): CER 0.1386, length ratio 0.949; the dominant errors are deletions (space 195, `/` 98, e 87, n 78, t 56, r 55 ...) against fewer insertions (space 99, `/` 57). The blank row has bias +0.998 against -1.58 for other rows: the model is over-eager to emit blank.

H1 (bias shift, O_18.lin.bias, blank): lowering the blank bias by 0.3 cuts deletions. Predicted CER -0.004 to -0.012 (to ~0.130). If it fails: deletions are not a calibration problem but a feature problem (the evidence is in the rows, not the bias); stop bias edits on the blank.
H2 (same, further -0.3, cumulative -0.6): only if H1 kept. Predicted a further -0.002 to -0.006, with diminishing returns and a risk of insertions. If it fails: optimum is near -0.3.
H3 (bias, space): raising the space bias by +0.3 trades 195 deletions against 99 insertions. Predicted -0.002 to -0.005. If it fails: space errors are a segmentation/context matter, not a prior one.
H4 (bias, `/`): `/` +0.3, deleted 98 vs inserted 57. Predicted -0.001 to -0.003; may be flat (the interval may contain 0). If it fails: `/` is bimodal (line-ending vs. word-separator use) and not fixable by a prior.
H5 (donor interpolation, all C_* and L_* tensors toward schlussredaktion-v1, alpha 0.25; O_18 skipped because codecs differ, rule_mask = unseen_by donor): the sibling shares the hertwig-v2 parent, so averaging is plausible and its unseen-line CER is the lower one of the two donors. Predicted -0.003 to -0.010; if it fails: the models are not linearly connected at this scale (or the sibling has no advantage on unseen lines); no further interpolation.
H6 (only if H5 kept): alpha 0.5 total. Predicted a further -0.002 to -0.005; if it fails, keep alpha 0.25.
H7 (final, bias refinement after the other edits): re-tune the blank bias by -0.15 on the final state, since the rows changed. Predicted -0.001 to -0.003.
Total expected gain if all hold: CER from 0.139 to about 0.12-0.128. No edits of the convolution stack on their own; no donor use for the biases.


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
