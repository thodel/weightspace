# Pre-registration — 20261113-planted-adapter-p3-alpha2-a-f1-b6-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (a), **Fable 5.1 rerun** (pack `q02`, attempt 2) of `runs/20261107-planted-adapter-p3-alpha2-a-b6-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-adapter-f1/q02/a/program.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 6 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

# Pre-registered hypotheses — q02, task (a), budget 6

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort as configured by the caller (not raised by me); tools used: Read (brief, pack, the four API files), Write (these two files), Bash only for `py_compile`. No pass run, no adapter loaded, nothing read outside the allowed files.

## What the pack shows

- `lora_alpha` = 256, r = 64, scale 4.0. Both siblings: alpha 128, scale 2.0.
- Every one of the 128 modules has ‖ΔW‖ / sibling-median between 1.60 and 1.85 (median ≈ 1.79); the total is 235 against 127 / 136. There is no module that stands out from that band in either direction; effective ranks and σ1 look like a normal depth profile (layer 0 and the late gate/q modules lower rank, mid layers ~30–48).
- C0: CER 2.50, 29 / 32 pages at the cap, 28 loops, and even the pages that did not overrun are at CER ≥ 0.77 — every page is bad, which points to a global defect, not a local one.
- Reading: the update the base receives is twice what the recipe produces (alpha doubled, 256 instead of 128; the B·A themselves are ≈0.9× the siblings, ordinary run-to-run variation). ratio ≈ 1.8 = 2 × 0.9.

## Hypotheses (one line each: edit class — prediction — if it fails)

- **H1 (main)** — `scale`, all 128 `lora_B` × 0.5 (= set alpha back to 128; config stays 256, the saved adapter is equivalent). Predicted: CER falls from ≈2.5 to below 0.25 (expected 0.05–0.15, i.e. −2.3 points of page CER with overrun), pages at the cap fall from 29 to ≤ 3. If refused by the keep rule: the defect is not (only) the global scale; nothing else in the pack is anomalous, so what remains is invisible to norm/rank statistics (a sign flip or permutation inside a module) and cannot be located with 4 passes — I would then try one gentler global factor (H1b) and otherwise freeze with no further edit.
- **H1b (only if H1 refused)** — `scale`, all `lora_B` × 0.7. Predicted: CER −0.5 to −1.5 if the over-scale is real but 0.5 overshot; if refused too: conclude the scale is not the defect, freeze.
- **H2 (data-driven, zero passes unless it fires)** — spectral outlier check against the two siblings computed inside the program (`directions()` per module on this adapter and on both siblings; per-module top-singular-value share σ1/‖S‖ and gap σ1/σ2). A module is flagged only if its share exceeds the siblings' mean by > 0.20 absolute and > 1.6× relative, or its gap is > 4 and > 3× the siblings' maximum. Edit: `add` on that module's B, delta = −(1−t)·u1(u1ᵀB), shrinking σ1 to the siblings' share (t = target/σ1), at most 3 modules in one edit. Predicted: if a module is flagged, CER −0.01 to −0.05 (a planted spike removed); I expect **nothing to be flagged** (the norm ratios are flat, a spike would have raised one of them). If flagged and refused: the outlier is natural variation, revert.
- **H3 (only if loops remain after H1: last kept state has ≥ 4 pages at the cap)** — `scale`, all `lora_B` × 0.8 (alpha 128 → ≈102). Predicted: small change, CER −0.02 to −0.1 and fewer pages at the cap; if refused: the residual loops are not scale-driven, the halved adapter is the final state. If kept and still ≥ 4 at the cap and a pass is left, one more step of 0.8 (same prediction); never more than two steps.

## Pass plan (≤ 6)

1. C0 baseline. 2. H1. 3–6. H2 if flagged (1 pass), H3 if loops remain (1–2 passes); H1b only in the branch where H1 was refused. Unused passes stay unused. Always ends with `freeze()`.


6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only runs/planted-adapter-f1/q02/pack/pack.md and runs/planted-adapter-f1/PROGRAMMER_BRIEF.md (the Sonnet brief with the model id and the read-restriction changed); not told which defect was planted; never shown the Sonnet runs or results. Operator: a Claude Opus 5.5 session that knows the defect and the Sonnet results and wrote no hypothesis or edit. Wrote its program in advance, did not run it.
