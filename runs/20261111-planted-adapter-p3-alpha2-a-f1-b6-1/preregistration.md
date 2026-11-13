# Pre-registration — 20261111-planted-adapter-p3-alpha2-a-f1-b6-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-adapter-p3-alpha2-a-b6-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-adapter-f1/q01/a/program.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 6 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only runs/planted-adapter-f1/q01/pack/pack.md and runs/planted-adapter-f1/PROGRAMMER_BRIEF.md (the Sonnet brief with the model id and the read-restriction changed); not told which defect was planted; never shown the Sonnet runs or results. Operator: a Claude Opus 5.5 session that knows the defect and the Sonnet results and wrote no hypothesis or edit. Wrote its program in advance, did not run it.
