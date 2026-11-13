# Pre-registration, pack xix-blb-v1, programmer a3

Model id: claude-sonnet-5-5. Subject kraken-german-xix-blb-v1; donor: parent kraken-german-xix-v1 (never trained on BLB; seen_by empty, so rule_mask is all-true, still passed via unseen_by).
Budget 10 passes (pass 1 = C0, CER 0.1383). Keep rule: harness default. Program uses at most 8 passes.
Evidence read: c0_practice.md: errors are dominated by deletions (space 78, e 44, i 29, n 26, r 23, t 21, s 21) against 30 space insertions; blank row has bias +1.232 and the smallest norm, space bias -7.85.

- H1 (add, O_18.lin.bias[blank] -= 0.4, then another -0.4 if kept, else -0.2): the child over-predicts the CTC blank on unseen handwriting, causing the deletions. Predicted CER change -0.005 to -0.015 (relative -4 to -10 %). If it fails (CER rises): deletions are not a global blank-prior issue (they are character-specific or from weak evidence); drop the blank-bias family.
- H2 (add, O_18.lin.bias[space] += 0.5): space is the top deleted character (78) with a very low bias (-7.85); a bias raise recovers some deletions at a cost of insertions (30). Predicted -0.002 to -0.006. If it fails: space deletions are a visual-evidence problem, not a prior.
- H3 (add, O_18.lin.bias for e,i,n,r,t,s,a,d,h += 0.3): the frequent-letter rows are the weakest (norm 7.6-8.3) and the most deleted; raising their bias targets the same under-emission. Predicted -0.002 to -0.008, partly overlapping H1. If it fails: the under-emission is not letter-frequency related.
- H4 (interpolate, all non-output tensors, alpha = -0.1 toward the parent, i.e. extrapolation away from it): the child's BLB adaptation moved weights from the parent along a direction that generalises to other BLB hands; extending it a bit further helps. Predicted -0.003 to -0.010 for alpha -0.1; if kept, try -0.2. If it fails: the adaptation is already at or past the useful point on unseen objects; weights are not usefully extrapolable.
- H5 (scale, O_18.lin.weight row 0 x 0.95, only if H1 was kept): shrinking the blank row logit (shape not just prior) further trims blank dominance. Predicted -0.001 to -0.004. If it fails: the blank prior is a pure bias effect.
Failure of H1 to H3 together would indicate that output-layer calibration is not the bottleneck, and the remaining error is feature-level (recurrent/conv).
