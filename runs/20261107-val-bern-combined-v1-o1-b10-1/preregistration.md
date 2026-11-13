# Pre-registration — 20261107-val-bern-combined-v1-o1-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/bern-combined-v1/o1/program.py` copied unchanged.

1. **Subject:** `kraken-bern-combined-v1.mlmodel` — tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`. Donors: `kraken-bern-hertwig-v2` (trained on all 6 Hertwig val pages: 212 lines seen) and `kraken-bern-schlussredaktion-v1` (trained on 8 of the 11 Schlussredaktion val pages: 216 lines seen); donor rule applied through `rule_mask`.
2. **Practice set:** `data/practice/bern-val.jsonl`, 513 lines, sha256 `30d2b536deef178ab36d07e85180fa003f45fc015e657cda79b3e6b06031dbf2`. Source: the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1 (pages_val.lst; partition 0.9, page level), 6 Hertwig pages from the 22.10 export (correct images), 11 Schlussredaktion pages. **Not training data of the subject** (audit `data/practice/overlap-bern-val.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines, sha256 `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

Model id: claude-opus-5-5 (programmer of attempt o1, subject kraken-bern-combined-v1)

# Pre-registration: weight-space session on kraken-bern-combined-v1 (validation practice, o1)

## Settings
- Subject: kraken-bern-combined-v1 (4.10 M params, 270 output rows, row 0 = CTC blank, bias +0.998).
- Practice: bern-val, 513 lines (212 Hertwig, 301 Schlussredaktion). C0: CER 0.1386, WER 0.4264, 27 collapsed lines,
  length ratio 0.949; errors are dominated by deletions (space→∅ 195, /→∅ 98, letters e/n/t/r/i/s→∅ 47–87 each) over insertions (∅→space 99, ∅→/ 57).
- Budget: 10 passes (pass 1 = C0 baseline), keep rule of the harness (paired 95 % interval of CER(after) − CER(last kept) below 0).
- Donors: kraken-bern-schlussredaktion-v1 (sibling, shares the hertwig-v2 parent; keep rule on `unseen_by('kraken-bern-schlussredaktion-v1')`, 297 lines),
  kraken-bern-hertwig-v2 (parent; keep rule on `unseen_by('kraken-bern-hertwig-v2')`, 301 Schlussredaktion lines).
- Own extra guard for donor edits: a donor edit is only offered to `s.keep` if CER on all 513 lines is also not higher than the last kept
  state's; otherwise it is reverted (the mask alone could hide damage on the lines the donor saw).
- Output-layer rows of donors are aligned by character (`wsp.align`); rows the donor lacks keep the subject's own values.
- No `logits` look-ahead on the practice set: every reading of it is a counted `measure`.

## Hypotheses (edit class; predicted direction and size of ΔCER vs last kept; conclusion if it fails)
- H1 (bias, no donor): the deletion surplus and the length ratio 0.949 mean the blank wins too many marginal frames; blank bias −0.5 →
  CER **down 0.5–1.0 pp** (0.1386 → ~0.130), fewer collapsed lines. If refused: the deletions are not decided at the blank/character margin
  (they are misreads of unseen hands, not calibration), and further bias work on the blank is pointless.
- H1b (bias, no donor, branch): if H1 kept, a further −0.5 (total −1.0) → CER **down a further 0.2–0.5 pp**; if H1 refused, −0.25 instead →
  **≈0 to −0.3 pp**. Fail: the optimum of the blank offset lies at the first kept step (or at 0).
- H2 (bias, no donor): space deletions exceed insertions 195:99; space bias +0.3 → CER **down 0.1–0.3 pp**. Fail: space errors are
  word-segmentation misreads of the image, not a prior that is too low.
- H3 (bias, no donor): '/' deletions exceed insertions 98:57; '/' bias +0.3 → CER **down 0.05–0.15 pp** (probably too small to pass the rule).
  Fail: the virgule is confused with other strokes, not under-predicted.
- H4 (interpolate, donor schlussredaktion-v1, all shared tensors, output rows by character, α = 0.25): the subject and its sibling are
  fine-tunes of the same parent, so a partial soup stays in one basin and averages out noise → CER on the masked 297 lines **down 0.3–1.0 pp**.
  Branch: kept → α a further 0.2 (**down ≤0.3 pp**); refused → α = 0.1 (**≈0 ± 0.3 pp**). Fail: the sibling (CER 0.182 on its unseen lines, vs
  0.1386 for the subject on all) is simply worse on those lines and averaging imports its errors; soups between these siblings do not help.
- H5 (bias, no donor): after whatever was kept, the blank offset is re-tuned by −0.25 → **≈0 to −0.2 pp**. Fail: calibration already at its optimum.
- H6 (interpolate with α < 0, donor hertwig-v2 = parent; task-vector extrapolation, α = −0.15): pushing further along the parent→subject
  fine-tuning direction improves the Schlussredaktion hand that the fine-tune added → CER on the 301 masked lines **down 0.2–0.6 pp**,
  with the full-set guard protecting the Hertwig lines. Fail: the fine-tune is at its end point; extrapolation leaves the basin (CER up ≥1 pp).

Expected overall: 1–3 edits kept, final CER on practice about 0.125–0.133; the bias edits (H1/H2) are the most likely to pass.


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Opus 5.5 (claude-opus-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
