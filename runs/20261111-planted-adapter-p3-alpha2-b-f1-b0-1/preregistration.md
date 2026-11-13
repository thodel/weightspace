# Pre-registration — 20261111-planted-adapter-p3-alpha2-b-f1-b0-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-adapter-p3-alpha2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-adapter-f1/q01/b/program.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 0 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — adapter q01, task (b), budget 0 (blind programmer)

- Model: claude-fable-5-1 (Claude Fable 5.1), run as a Claude Code subagent.
- Settings: reasoning effort as set by the harness (not shown to me; no explicit override). Tools used: Read (brief, pack,
  wsp/ops.py, wsp/session.py, wsp/run.py, wsp/lora_subject.py), Bash only for `ls`/`wc` on allowed files and
  `python -m py_compile` of my program, Write for the two files. No pass, no adapter loaded, nothing else read.

## What the pack shows

1. **Config:** `lora_alpha = 256`, `r = 64`, scale 4.0. Both siblings (same base, same recipe): `lora_alpha = 128`, scale 2.0.
   The alpha is exactly doubled.
2. **Norms:** every one of the 128 modules has ‖ΔW‖ / sibling-median between 1.60 and 1.85 (median ≈ 1.78); the total is
   234.99 vs 127.35 / 135.65 (×1.73–1.85). The excess is global and flat across layers and module kinds: no module, layer
   or kind stands out. A uniform factor of ~2 (times the normal ~0.9 run-to-run variation; the two siblings themselves differ
   by 6.5 %) is exactly what a doubled alpha on an otherwise healthy adapter produces.
3. **No local defect visible:** effective ranks are 16–48 (none near 64, so no injected noise; none near 1, so no rank
   collapse); σ1 and ‖B‖, ‖A‖ vary smoothly with depth; the few lower ratios (layers.3.q_proj 1.60, layers.23.q_proj 1.68)
   are within the spread. Zeroed rows/columns, a blanked module, or added noise would show as a ratio or effrank outlier,
   and none does.
4. **C0 reading:** page CER 2.50 (250 %), length ratio 2.55, **29 of 32 pages at the 3,072-token cap, 28 looping**;
   truncated CER 0.82. This is a decoder driven into repetition, the typical symptom of an over-strong adapter, not of a
   locally broken one (which gives bad characters, not runaway length on 90 % of pages).

## Hypothesis H1 — the adapter's scale is doubled (lora_alpha 256 instead of the recipe's 128)

- **Edit class:** global `scale` of every `lora_B.weight` (all 128 modules) by **0.5**. Per the brief, scaling every B by f is
  a change of lora_alpha by f; the saved adapter keeps `lora_alpha = 256` in its config and its update becomes
  ΔW = 2·B·A, the siblings' scale. After the edit the per-module ratio to the sibling median should sit at ≈ 0.9, within the
  siblings' own spread.
- **Predicted effect:** large improvement. Pages at the cap from 29 to a handful (≤ 8 of 32); loops from 28 to ≤ 5;
  page CER from 2.50 to well below 0.3 (most likely 0.05–0.15, i.e. a working transcription model); truncated CER from
  0.82 to the same level as the page CER.
- **Why I am fairly sure:** three independent readings (config vs siblings, a flat ×1.8 norm excess over all 128 modules,
  catastrophic looping at C0) agree on one cause. Risk is asymmetric: if the config were honest and the adapter had really
  been trained at alpha 256, halving it would give a half-strength adapter, which degrades gently; an over-scaled adapter
  loops, which is what C0 shows.
- **Alternative considered and rejected:** scale by 1/1.78 ≈ 0.56 to match the sibling median exactly. The recipe's alpha is
  128, and the siblings differ from each other by several per cent; 0.5 is the principled correction, 0.56 would fit noise.
- **In-program guard (no pass needed):** the program loads the two siblings as donors (if reachable in the sandbox) and
  compares this adapter's *unscaled* ‖B·A‖ per module with the sibling median. If the median of those ratios lies in a broad
  natural band (0.5–1.6), the excess in the pack is the config's alone and the edit goes through as planned. If the siblings
  cannot be loaded, the pack's evidence is sufficient and the edit goes through anyway. Modules whose unscaled ratio falls
  outside 0.4–2.5× the global median are noted as suspects but **not edited**: I am not sure enough of any local repair
  without a pass.

## What I do not apply

- No per-module rescaling, transplant, projection or row/column repair: the pack shows no local anomaly, and a wrong local
  edit at budget 0 is uncorrectable.
- No change to A matrices (independently initialised; nothing from the siblings can be mixed in).


6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only runs/planted-adapter-f1/q01/pack/pack.md and runs/planted-adapter-f1/PROGRAMMER_BRIEF.md (the Sonnet brief with the model id and the read-restriction changed); not told which defect was planted; never shown the Sonnet runs or results. Operator: a Claude Opus 5.5 session that knows the defect and the Sonnet results and wrote no hypothesis or edit. Budget 0.
