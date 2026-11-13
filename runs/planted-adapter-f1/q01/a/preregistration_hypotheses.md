# Pre-registration — planted-adapter-f1 / q01 / task (a), budget 6

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort; tools used: Read/cat of the pack,
the brief and `wsp/{ops,session,run,lora_subject}.py`, Write, and `py_compile` of `program.py`. No pass run, no adapter loaded.

## Reading

- Config: `lora_alpha = 256`, `r = 64` → scale 4.0. Both siblings (same base, same recipe): `lora_alpha = 128`, scale 2.0.
- Every one of the 128 modules has ‖ΔW‖ ratio to the sibling median in 1.60–1.85 (median ≈ 1.78), with no outlier module
  up or down; total ‖ΔW‖ 235 vs 127/136 (×1.8). The uniformity says: one global factor, not a per-module defect. Ratio ≈ 1.8 rather
  than 2.0 means this run's intrinsic ‖B·A‖ is ≈ 0.9× the siblings' — ordinary run-to-run variation, not a second defect.
- Effective ranks, σ1/‖ΔW‖, ‖A‖ and ‖B‖ per kind look homogeneous across layers (no zeroed module, no rank-1 spike, no single
  module with an inflated B or A).
- C0: CER 2.50 (length ratio 2.55, 29/32 pages at the 3,072-token cap, 28 loops; truncated CER 0.82; even non-capped pages ≈ 0.77–1.0).
  The adapter is unusable as it is, consistent with an update driven at twice its trained strength.

## Hypotheses (one line each: edit class — prediction — if it fails)

- **H1 (primary): the planted defect is a doubled scale (lora_alpha 128 → 256).** Edit class: `scale`, all 128 `lora_B` tensors × 0.5
  (equivalent to restoring alpha = 128; the config is copied unchanged on save, so the fix lives in B). Prediction: CER falls from 2.50 to
  below 0.20 (most likely 0.05–0.15), pages at the cap from 29 to ≤ 3, loops essentially gone; a very large, unmistakable drop far above
  the engine's noise. If it fails (keep refused): the global magnitude is not the defect, or not the only one; I re-apply the same edit once
  (H1r, the engine is not repeatable) and, if refused again, freeze the adapter unchanged and conclude that the defect is not visible in
  the norm statistics (sign/direction class) and cannot be targeted from this pack.
- **H2 (α-sweep, only after H1 is kept): the right factor is 0.45, not 0.5.** Edit class: `scale`, all `lora_B` × 0.9 on top of H1
  (effective scale 1.8, module norms at the siblings' median). Prediction: change within noise, |ΔCER| < 0.02, direction undecided; I expect
  the keep rule to refuse. If refused: the residual ×0.9 is this run's own norm, 0.5 is the whole correction.
- **H2b (α-sweep, only if H2 was refused): factor 0.55.** Edit class: `scale`, all `lora_B` × 1.1 on top of H1. Prediction: within noise,
  |ΔCER| < 0.02; expect refusal. If refused: the optimum is flat around 0.5 and the ×0.5 state is frozen as the repaired adapter.

## Controls

C0 (pass 1) is the baseline. H2/H2b together are a two-point α-sweep (C3) around the repaired scale. Budget: at most 4 of 6 passes
(C0, H1, H2, H2b) on the success branch; at most 3 (C0, H1, H1r) on the failure branch. Nothing else is edited: the readings show
no per-module defect to target, and an edit I cannot motivate from the pack would only add risk.
