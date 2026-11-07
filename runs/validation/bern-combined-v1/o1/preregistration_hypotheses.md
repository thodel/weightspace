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
