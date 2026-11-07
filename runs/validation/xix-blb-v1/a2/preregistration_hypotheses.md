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
