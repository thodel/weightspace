# Pre-registration — 20261107-planted-k10-noise-lstm14-60pct-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k10-noise-lstm14-60pct/a/program.py`, copied unchanged.

1. **Subject:** `k10-noise-lstm14-60pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 60 % of the block's; `runs/planted/plant.json`) — tensor digest `930c5c2b0279e885f4e87954711d76aaee3e9d07128fc217156821da237eb2fc`, file sha256 `c55386f265d20e1ba0952391b63ff5e071a8277ec4544cbe9b0a86e9453f2bad`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5; reasoning effort low; tools: Read, Bash (read-only listing, py_compile), Write. No pass run.

Reading: C0 CER 0.120 with 11 collapsed lines; top confusions are deletions (space, e, n, r, i) -- the output under-emits, i.e. diffuse damage, not a single wrong row. In the parent diff, L_14 (both directions) is far from the parent (rel L2 0.55 hh / 0.75 ih, cosine 0.79-0.87) while L_12 / L_16 sit at 0.2-0.3; C_3.bias is also off (rel 1.09) but is tiny. Suspicion: random noise added to L_14 weights.

H1 (interpolate L_14 weights, all 4 matrices, alpha=0.4 toward parent): predict CER 0.120 -> ~0.095 (-0.02 to -0.03). If it fails (CER up or no significant gain), the parent L_14 is not compatible with the child's neighbours at this mix; try smaller alpha 0.2 (H1b).
H2 (if H1 kept: a further step to alpha 0.5 of the remainder, total ~0.7): predict a further -0.005 to -0.015. If fails, revert and stay at H1.
H3 (if H1 fails entirely: scale L_14 weights by 0.9): noise inflates norm; predict -0.005. If fails, the L_14 deviation is structured (genuine fine-tuning), not noise.
H4 (interpolate C_3.co.bias toward parent, alpha 0.5): predict ~0 to -0.005 (tensor tiny). Expected to fail the keep rule; fine, it is cheap.
H5 (only if all above fail or a gain remains: interpolate L_14 biases hh, alpha 0.5): predict < 0.003 change. Conclude that bias is not damaged.
Overall: predicted final CER ~0.085-0.10. If H1 and H1b both fail, conclude L_14 is not the (only) defect and ship nothing.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
