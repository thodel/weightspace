# Pre-registration — 20261107-val-xix-blb-v1-a2-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/xix-blb-v1/a2/program.py` copied unchanged.

1. **Subject:** `kraken-german-xix-blb-v1.mlmodel` — tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`, file sha256 `48caacb4a742b5b5ccd2a92a005a4ca701bdf7cef7859087832b6d726f5f8df4`. Donors: `kraken-german-xix-v1` (parent; trained on no BLB data: `seen_by` is empty for every line).
2. **Practice set:** `data/practice/blb-val-300.jsonl`, 300 lines, sha256 `0118e9b73a3dd88add8d009236352f7c0dcb6ba5b4f81a619caff79b1baa8bca`. Source: eval-side objects of lassberg-blb-gt (the trainer's validation set for this model), 7 whole objects drawn from blb-eval-hand; the training side has no validation part (PHASE note in CONTROLS_REPORT §3). **Not training data of the subject** (audit `data/practice/overlap-blb-val-300.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/blb-eval-hand-b.jsonl`, 1718 lines, sha256 `078b208a4e104d4ce165e7d21fdcd963062f308faa6389911a78c09f87116a4f`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, xix-blb-v1, programmer a2

Model id: claude-sonnet-5-5. Subject kraken-german-xix-blb-v1 (CER 0.1383 on blb-val-300). Donor: parent kraken-german-xix-v1 (CER 0.3266; seen_by empty, so rule_mask is a no-op but passed anyway). Budget 10 passes, pass 1 = C0. min keep rule as the harness: paired 95 % interval below 0. Edits are applied cumulatively when kept.

Observation driving everything: C0 errors are dominated by deletions (space->nothing 78, e 44, i 29, n 26, r 23, t 21, s 21) against 30 insertions of space; blank bias is +1.23 vs -1.92 mean for characters. The fine-tune moved the model away from the parent (rel L2 about 0.14-0.3 in LSTMs), toward handwriting.

H1 (add, O_18.lin.bias row 0 -= 0.5): the model under-emits; lowering blank bias recovers deleted characters. Predict CER -0.005 to -0.015 (0.138 -> ~0.127). If it fails (CER up): deletions come from weak evidence not blank over-confidence; the opposite sign gets one try (+0.5) because 2 lines are collapsed; otherwise drop the blank-bias family.
H2 (follow-on, further -0.5 if H1 kept, else +0.5 if H1 was refused with CER up): predict a smaller gain (-0.002 to -0.006) or a loss, locating the optimum near the first step.
H3 (add, O_18.lin.bias space += 1.0): space is the largest deletion class (78) but also 30 insertions; its bias is -7.85. Predict -0.004 to -0.010. If it fails, space errors are position errors, not threshold errors.
H4 (interpolate with donor, alpha = -0.25, all conv + LSTM tensors, i.e. W + 0.25 (W - parent)): task-vector extrapolation; the fine-tune direction toward handwriting generalizes to unseen hands, so amplifying it helps. Predict -0.005 to -0.02. If it fails, the fine-tune is already at/over its optimum for new hands; follow-up tries the opposite sign (alpha +0.15, shrink toward the parent), predicting +0.01 or worse (parent is far worse).
H5 (add, O_18 blank row weight scaled 0.9): weakens blank evidence in the same way as H1 but input-dependent. Predict -0.002 to -0.008 over the kept state, and smaller if H1/H2 already took the gain.
H6 (add, bias += 0.3 on e,i,n,r,t,s): the frequent deleted letters; predict -0.002 to -0.006, overlapping with H1. If it fails, deletions are not a global prior-level effect.
Remaining passes: confirm combined state or one further step along the best-working direction. Nothing here predicts a gain above 0.03 CER; a result near 0 means the subject is near a local optimum for this practice set.


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
