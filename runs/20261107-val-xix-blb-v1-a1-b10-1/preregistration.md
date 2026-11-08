# Pre-registration — 20261107-val-xix-blb-v1-a1-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `/home/tobias/Repo/weightspace/.claude/worktrees/article-controls-8803eb/runs/validation/xix-blb-v1/a1/program.py` copied unchanged.

1. **Subject:** `kraken-german-xix-blb-v1.mlmodel` — tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`, file sha256 `48caacb4a742b5b5ccd2a92a005a4ca701bdf7cef7859087832b6d726f5f8df4`. Donors: `kraken-german-xix-v1` (parent; trained on no BLB data: `seen_by` is empty for every line).
2. **Practice set:** `data/practice/blb-val-300.jsonl`, 300 lines, sha256 `0118e9b73a3dd88add8d009236352f7c0dcb6ba5b4f81a619caff79b1baa8bca`. Source: eval-side objects of lassberg-blb-gt (the trainer's validation set for this model), 7 whole objects drawn from blb-eval-hand; the training side has no validation part (PHASE note in CONTROLS_REPORT §3). **Not training data of the subject** (audit `data/practice/overlap-blb-val-300.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/blb-eval-hand-b.jsonl`, 1718 lines, sha256 `078b208a4e104d4ce165e7d21fdcd963062f308faa6389911a78c09f87116a4f`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, xix-blb-v1, programmer a1 (model: claude-sonnet-5-5)

Settings: practice set blb-val-300, budget 10 passes (pass 1 = C0), no donor mask needed for non-donor edits; the parent donor edit uses rule_mask=s.unseen_by("kraken-german-xix-v1") (seen_by is empty, so all lines).
Basis: C0 CER 0.1383, errors dominated by deletions (space->nothing 78, e 44, i 29, n 26, r 23, t 21, s 21), length ratio 0.97; blank row is the weakest (norm 4.6) with bias +1.232, the only positive bias besides s/E/d/1; space bias -7.85 is extreme.

H1 (O_18.lin.bias, add): lowering the blank bias by 0.5 (blank over-predicted) lowers CER by ~0.005-0.015 (relative 5-10 %). If it fails: deletions are not a calibration problem of the blank logit; drop the blank-bias line.
H2 (O_18.lin.bias, add, conditional on H1 kept): a further -0.5 gives a smaller extra gain (~0.002-0.006). If it fails: the optimum is already reached at -0.5.
H3 (O_18.lin.bias, add on space): space bias -7.85 is too low (space->nothing is the largest confusion); raising it by +1.0 lowers CER by ~0.003-0.010. If it fails: space deletions come from features, not from the prior; but insertions (30 extra spaces) may be the reason.
H4 (O_18.lin.bias, add on e,i,n,r,t,s,a with the weakest rows): +0.4 on their biases lowers CER by ~0.002-0.006. If it fails: the deletions of frequent letters are blank dominance, already handled by H1.
H5 (O_18.lin.weight, scale blank row): the blank row has low norm (4.6), so shrinking it by x0.9 reduces blank confidence ~ +/- 0.003 (uncertain). If it fails: blank bias is the cleaner knob.
H6 (L_* , interpolate toward parent, alpha 0.15 on all LSTM tensors): the parent is much worse here (CER 0.327) and the subject was trained away from it, so I expect no gain or a rise of 0.00-0.02. Included as a tested control of "parent as regulariser"; if kept the conclusion is that fine-tuning overshot to BLB. If it fails (expected): the donor carries no usable information for this domain.


6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it.
