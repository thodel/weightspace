# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read and shell (py_compile only), no model loaded, no passes.

Evidence: C0 CER 0.199 with errors almost all deletions (reference char -> nothing, length ratio 0.83, 24 collapsed lines): the blank wins too often, i.e. logits are under-confident. In the diff, L_16 (all 8 tensors, including bias_hh) is at rel L2 0.39-0.55 vs the parent with cosine 0.95-0.999, and its rms is about half of L_12/L_14's (bias_hh rms 0.42 vs 0.92); bias_hh cosine 0.999 with rel 0.505 is the signature of a pure uniform scale by 0.5. The other blocks differ from the parent by 0.1-0.3 with no such signature.

| # | edit class | hypothesis | predicted CER change | if it fails |
|---|---|---|---|---|
| H1 | scale L_16 (all 8 tensors) x2.0 | A x0.5 shrink was planted on the last BiLSTM; it starves the output layer, so the blank wins and characters are deleted | 0.199 -> about 0.08-0.12 (delta -0.08 or more), deletions of ` `,e,n,i fall | If CER does not fall: no uniform scaling on L_16; try H1b |
| H1b | scale L_16 weights only (not biases) x2.0, tried only if H1 is refused | gate biases were not touched, only weights | delta -0.05 | scaling is not the defect; go to H3, H4 |
| H1c | scale L_16 by a further x1.15 or x0.9 on top of H1 (best of at most two) | the true factor is not exactly 2 because of fine-tuning drift | delta of -0.005 to -0.01, or nothing | stop refining; keep H1 as is |
| H3 | add to O_18.lin.bias[0] (blank) -0.5, then -1.0 if kept | residual blank over-preference after H1 | delta -0.005 to -0.02 | the blank bias is healthy |
| H4 | interpolate C_3.co.bias toward the parent (alpha 0.5) | C_3.co.bias is far from parent (rel 1.09, cos 0.68), possibly a planted noise; but its rms is tiny (0.018), so little effect | delta around 0 (|delta| < 0.003); expected refused | C_3 bias is natural drift; leave |

Conclusion rules: a defect is confirmed only when s.keep accepts (paired 95 % interval below 0). If the baseline-to-H1 gain is under 0.02 the L_16 scale hypothesis is rejected and the deletion pattern is attributed to something else (blank row or bias); no further guess is shipped.
Measure budget: C0, H1, [H1b], H1c (<=2), H3 (<=2), H4 = at most 8 passes plus spare.
