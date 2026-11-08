# Pre-registration — 20261107-val-xix-blb-v1-o1-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/xix-blb-v1/o1/program.py` copied unchanged.

1. **Subject:** `kraken-german-xix-blb-v1.mlmodel` — tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`, file sha256 `48caacb4a742b5b5ccd2a92a005a4ca701bdf7cef7859087832b6d726f5f8df4`. Donors: `kraken-german-xix-v1` (parent; trained on no BLB data: `seen_by` is empty for every line).
2. **Practice set:** `data/practice/blb-val-300.jsonl`, 300 lines, sha256 `0118e9b73a3dd88add8d009236352f7c0dcb6ba5b4f81a619caff79b1baa8bca`. Source: eval-side objects of lassberg-blb-gt (the trainer's validation set for this model), 7 whole objects drawn from blb-eval-hand; the training side has no validation part (PHASE note in CONTROLS_REPORT §3). **Not training data of the subject** (audit `data/practice/overlap-blb-val-300.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/blb-eval-hand-b.jsonl`, 1718 lines, sha256 `078b208a4e104d4ce165e7d21fdcd963062f308faa6389911a78c09f87116a4f`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

Model id: claude-opus-5-5

# Pre-registered hypotheses — kraken-german-xix-blb-v1 (attempt o1)

## Settings
- Subject: `kraken-german-xix-blb-v1` (fine-tune of `kraken-german-xix-v1` on BLB handwriting; codec grew 224 -> 296, shared order preserved).
- Donor: `kraken-german-xix-v1` (parent), CER 0.3266 on the practice set; `seen_by` empty for every line, so every donor edit is judged with `rule_mask=s.unseen_by("kraken-german-xix-v1")` (all 300 lines).
- Practice: blb-val-300 (300 handwritten lines, 7 objects never trained on). C0: CER 0.1383, length ratio 0.970, 2 collapsed lines.
- Budget: 10 passes, pass 1 = C0 baseline. Worst-case plan uses 9. Keep rule: harness's paired 95 % interval below 0; `s.keep()` decides every edit.

## Reading of the pack
- Errors are dominated by deletions: space->∅ 78, e/i/n/r/t/s->∅ 21-44 each, against only 30 spurious space insertions; length ratio 0.970. The model under-emits.
- The blank row has bias +1.23 (only strongly positive bias) and the most frequently deleted letters (e, r, s, t, i, n) have the weakest output rows (norm 7.6-8.3), so the blank wins frames where these thin letters are weakly evidenced. Space bias is the most negative of all rows (-7.85).
- The parent is far worse on this set (0.327 vs 0.138), so the fine-tune direction (child - parent) is useful here; moving toward the parent should hurt, moving further along the direction might help a little.

## Hypotheses (one edit class each; decisions by the keep rule)
- H1 (bias, blank): lowering `O_18.lin.bias[blank]` by 0.5 converts some deletions into correct emissions. Predicted CER change -0.3 to -1.0 points (to ~0.130-0.135); if kept, a further -0.5 (cumulative -1.0) gives another -0.2 to -0.5 points, and a third -0.5 is tried only if the second was kept. If the first step raises CER: deletions are not a blank-margin problem (the evidence for the letters is missing upstream), and the blank bias is left alone. If it falls but not significantly, one direct try at -1.0 total.
- H2 (bias, space): raising the space bias by +0.5 recovers some of the 78 space deletions at the cost of fewer than half as many new spurious spaces. Predicted CER -0.1 to -0.4 points; a second +0.5 only if kept. If it fails: space deletions are true ambiguity in handwritten word spacing (inter-word gaps similar to intra-word gaps), not a calibration offset.
- H3 (donor, task-vector extrapolation): `interpolate` with the parent at alpha = -0.1 on all 34 tensors (output rows aligned by character; the 72 child-only rows untouched) amplifies the fine-tune direction by 10 %. Predicted CER -0.1 to -0.5 points. If CER rises, the fine-tune is already at/over its optimum for unseen BLB hands, and the opposite move (alpha = +0.05, WiSE-FT style regularisation toward the parent) is tried once, predicted 0 to -0.3 points; if both fail, the subject sits at a local optimum along the parent-child line.
- H4 (bias, letters): +0.3 on the biases of e, i, n, r, s, t (the most-deleted letters with the weakest rows). Predicted CER -0.1 to -0.4 points. If it fails (or substitutions such as r->n, r->e grow): their deletions are recognition failures, not calibration, and no output-layer bias can fix them.

Expected overall: final practice CER 0.128-0.136 (-0.2 to -1.0 points vs C0), driven mainly by H1; held-out gain smaller than practice gain.


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Opus 5.5 (claude-opus-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
