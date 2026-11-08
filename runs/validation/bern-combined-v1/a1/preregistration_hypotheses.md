# Pre-registration, bern-combined-v1, programmer a1

Model id: claude-sonnet-5-5. Subject kraken-bern-combined-v1; donor: kraken-bern-schlussredaktion-v1 (sibling, hidden tensors only, never O_18 because codecs differ). Budget 10 passes (pass 1 = C0). Baseline from pack: CER 0.1386, length ratio 0.949, 27 collapsed lines, errors dominated by deletions (space 195, "/" 98, e 87, n 78) versus fewer insertions (space 99, "/" 57). Keep rule: harness default (paired 95 % interval below 0). Donor edits judged with rule_mask=s.unseen_by(schlussredaktion); once such an edit is kept, all later passes use the same mask.

- H1 (pass 2) add_bias: O_18.lin.bias[blank] -= 0.3. The model under-emits (ratio 0.949, deletions >> insertions), blank bias is +1.0 and blank row is the weakest. Prediction: CER falls 0.003 to 0.010 (to about 0.130-0.135). If it fails (CER rises): blank is not over-dominant; deletions come from low character evidence, not the blank prior; drop blank-bias route (try +0.15 at most, otherwise abandon).
- H2 (pass 3) add_bias: a second step of the same size if H1 kept (total -0.6), else a smaller -0.15. Prediction: further gain of 0.001-0.004, saturating, insertions begin to rise. If it fails: optimum is near the first step.
- H3 (pass 4) scale on O_18.lin.weight blank row by 0.9 (lowers blank logit magnitude where features align). Prediction: 0.000 to -0.003; likely not significant. If it fails: the blank effect is in the bias, not the row.
- H4 (pass 5) interpolate hidden tensors (C_*, L_*) toward schlussredaktion-v1 with alpha 0.25 (both siblings share the hertwig-v2 ancestor, so linear mode connectivity is plausible; donor CER on all lines 0.118 vs subject 0.139, but 0.182 on its unseen lines). Prediction: CER -0.002 to -0.008 on the donor-unseen lines; risk of rise if tensors have drifted apart. If it fails: the siblings are not in one basin at this distance; no further donor edits.
- H5 (pass 6) if H4 kept, extrapolate total alpha to 0.5. Prediction: further -0.001 to -0.005 or a rise (the curve is expected to be convex; a rise means the optimum is below 0.5). If it fails: keep alpha 0.25.
- H6 (pass 7) add_bias: space bias +0.2 (195 deleted spaces vs 99 inserted). Prediction: -0.002 to -0.006 (word-level errors). If it fails: space errors are alignment/segmentation errors, not prior errors.
- H7 (pass 8) add_bias: "/" bias +0.2 (98 deleted vs 57 inserted). Prediction: -0.001 to -0.003. If it fails: "/" confusion is evidence-driven.
- H8 (pass 9) add_bias on a +0.1 lift of the 6 most frequent lowercase letters (e n t r i s). Prediction: -0.001 to -0.004. If it fails: no per-letter prior adjustment helps; stop.
- Pass 10 is held in reserve and unused if nothing is left to test; final state is the last kept one.

Expected overall: 0.138 to 0.125-0.132, most of it from H1/H4. The held-out gain is expected smaller than on practice for bias edits fitted to the practice set.
