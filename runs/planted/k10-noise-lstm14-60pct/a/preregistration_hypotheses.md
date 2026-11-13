# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5; reasoning effort low; tools: Read, Bash (read-only listing, py_compile), Write. No pass run.

Reading: C0 CER 0.120 with 11 collapsed lines; top confusions are deletions (space, e, n, r, i) -- the output under-emits, i.e. diffuse damage, not a single wrong row. In the parent diff, L_14 (both directions) is far from the parent (rel L2 0.55 hh / 0.75 ih, cosine 0.79-0.87) while L_12 / L_16 sit at 0.2-0.3; C_3.bias is also off (rel 1.09) but is tiny. Suspicion: random noise added to L_14 weights.

H1 (interpolate L_14 weights, all 4 matrices, alpha=0.4 toward parent): predict CER 0.120 -> ~0.095 (-0.02 to -0.03). If it fails (CER up or no significant gain), the parent L_14 is not compatible with the child's neighbours at this mix; try smaller alpha 0.2 (H1b).
H2 (if H1 kept: a further step to alpha 0.5 of the remainder, total ~0.7): predict a further -0.005 to -0.015. If fails, revert and stay at H1.
H3 (if H1 fails entirely: scale L_14 weights by 0.9): noise inflates norm; predict -0.005. If fails, the L_14 deviation is structured (genuine fine-tuning), not noise.
H4 (interpolate C_3.co.bias toward parent, alpha 0.5): predict ~0 to -0.005 (tensor tiny). Expected to fail the keep rule; fine, it is cheap.
H5 (only if all above fail or a gain remains: interpolate L_14 biases hh, alpha 0.5): predict < 0.003 change. Conclude that bias is not damaged.
Overall: predicted final CER ~0.085-0.10. If H1 and H1b both fail, conclude L_14 is not the (only) defect and ship nothing.
