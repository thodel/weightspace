# Pre-registration — 20261113-planted-adapter-p3-alpha2-b-f1-b0-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (b), **Fable 5.1 rerun** (pack `q02`, attempt 2) of `runs/20261107-planted-adapter-p3-alpha2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-adapter-f1/q02/b/program.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 0 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — q02, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1. Settings: default reasoning effort (extended thinking on), tools used: Read, Bash (ls/wc on the allowed
files, `py_compile` on my program), Write. Read only: `PROGRAMMER_BRIEF.md`, `pack/pack.md`, `wsp/ops.py`, `wsp/session.py`,
`wsp/run.py`, `wsp/lora_subject.py`. Nothing in `a/`, nothing in `q01/`, no adapter loaded, no pass.

## What the pack shows

- `lora_alpha` = 256 at r = 64, scale 4.0. Both siblings (same base, same recipe): `lora_alpha` 128, scale 2.0.
- Total ‖ΔW‖ 234.99 against 127.35 / 135.65 for the siblings (×1.73–1.85).
- Every one of the 128 modules has ratio 1.60–1.85 to the sibling median (mean ≈ 1.78); no module stands out above the
  rest, and ‖B‖, ‖A‖ are uniform within each kind. This is a global factor on the update, not a module-level fault:
  with the config at 128 this adapter's own weights would sit at ≈ 0.89 of the siblings, inside their own spread (127 vs 136).
- C0: CER 2.50, 29 of 32 pages at the 3,072-token cap, 28 looping, length ratio 2.55 — the model writes past the page on
  almost every page. An update at twice its trained strength does exactly that.

## H1 (applied): the update is doubled — `lora_alpha` 256 where the recipe says 128

- Edit class: global scale. `scale` with factor 0.5 on all 128 `lora_B.weight` tensors (one edit). Scaling every B by 0.5 is
  the same as setting `lora_alpha` back to 128; the config file itself is copied unchanged by the harness, so the fix must
  live in the weights. 0.5 is exact in bf16, so nothing is lost on save.
- Predicted effect on the practice set: CER from 2.50 down to the range of a healthy page adapter, ~0.05–0.15 (a drop of
  ≥ 2 CER points absolute); pages at the cap from 29 to a handful (≤ 3), loops essentially gone; length ratio back near 1.
- If wrong (the adapter had really been trained at 256): the update is then half its trained strength, which costs a few
  CER points at most — the model stays a transcriber. The readings (same recipe, uniform ratio, catastrophic looping)
  make that reading unlikely; the asymmetry of the two outcomes is why this is the one edit I ship blind.
- Guards in the program before the edit: config must read `lora_alpha` 256, r 64, 128 modules; the total ‖ΔW‖ computed
  from the live tensors (with the adapter's scale) must be within 10 % of the pack's 234.99. If a guard fails the subject
  is not the one the pack describes and the program applies nothing.

## H2 (not applied): `layers.27.self_attn.q_proj` — effrank 17.0, σ1 16.5

Lowest effective rank of any q_proj and the largest σ1 among the attention modules, but its norm ratio (1.81) is exactly
in line with the rest, so no energy was added; ~36 % of its energy in the top direction is not out of range for the late
full-attention layers (layer 31 q_proj: σ1 11.0, effrank 28). Possibly natural. Blind, I would not project a direction out
of it; too likely to damage a healthy module.

## H3 (not applied): `layers.3.self_attn.q_proj` ratio 1.60, `layers.23.self_attn.q_proj` 1.68

Mildly below the ~1.78 mean, i.e. at alpha 128 these modules sit at ~0.80–0.84 of the siblings instead of ~0.89. Within what
run-to-run variation can do; and a shortfall of 10 % in one module is not something to correct with a sibling's statistics
blind. Nothing applied.

## What the program does

1. `open_session(".")`, guards above.
2. `s.hypothesis(H1)`, one `s.edit("scale", all_128_lora_B, factor=0.5)`, `s.keep_unmeasured(...)`.
3. Notes on H2/H3 (not applied), `s.freeze()`.


6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only runs/planted-adapter-f1/q02/pack/pack.md and runs/planted-adapter-f1/PROGRAMMER_BRIEF.md (the Sonnet brief with the model id and the read-restriction changed); not told which defect was planted; never shown the Sonnet runs or results. Operator: a Claude Opus 5.5 session that knows the defect and the Sonnet results and wrote no hypothesis or edit. Budget 0.
