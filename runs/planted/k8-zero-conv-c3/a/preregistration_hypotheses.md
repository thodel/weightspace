# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read/shell only, no passes used while writing.

Reading: C_3.co.weight and bias are exactly zero (rel L2 vs parent 1.0), so all downstream features are bias-only constant; C0 CER 1.000, every line empty. Everything else sits 0.13-0.33 from the parent, i.e. ordinary fine-tuning drift.

- H1 (main): C_3 was zeroed. Edit class: transplant C_3.co.weight + C_3.co.bias from the parent (kraken-german-xix-v1). Predict CER from 1.00 to about 0.3-0.6 (child C_0/C_6 drifted from parent, so mismatch remains). If it fails (CER >= 0.9 or not kept): the zeroing is not the only fault or the parent donor's feature space is too far; conclude downstream convs must co-adapt, go to H2.
- H2: the child's C_6/C_9 co-adapted to a C_3 we cannot recover; moving them toward the parent restores a consistent conv stack. Edit class: interpolate C_6 (w,b) toward parent alpha 0.5, then C_9 likewise. Predict a further 0.05-0.2 absolute CER drop. If it fails (not kept): keep child C_6/C_9, the mismatch is tolerable.
- H3: if H2 helped with alpha 0.5, full transplant (alpha 1) of C_6 and C_9 helps more; predict a further 0.02-0.1 drop; if not kept, stay at 0.5.
- H4: C_3 donor magnitude is off for the child's C_6 input: scale C_3 weight by 0.85 or 1.15. Predict at most 0.02 change; likely not kept (interval contains 0). Concluding: leave as is.
- H5: if output is still heavily collapsed (collapsed > 0), blank bias is too high relative to the child's drift: add -0.5 to O_18.lin.bias row 0. Predict small (0.01-0.05) improvement if blanks dominate deletions; else revert.
