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
