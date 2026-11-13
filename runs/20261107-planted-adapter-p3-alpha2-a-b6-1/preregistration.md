# Pre-registration — 20261107-planted-adapter-p3-alpha2-a-b6-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (a). Written before the first pass; the program is `runs/planted-adapter/p3-alpha2/a/program.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 6 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: low reasoning effort; tools Read, Write, Bash (py_compile only). Task (a), budget 6.

Evidence: lora_alpha 256 vs siblings 128 (scale 4.0 vs 2.0); every module's norm ratio is a uniform 1.7-1.85, total 235 vs 127-136; no module stands out. C0 CER 2.50 with 29/32 pages at the cap and 28 loops.

H1 (scale, all 128 B x0.5, i.e. alpha 256 -> 128). Predict CER falls from ~2.5 to ~0.3-0.6 and at_cap falls from 29 to under 10. If it fails (not kept), the update is not simply over-scaled; try a milder x0.7 and conclude the damage is not a pure global gain.
H2 (after H1 kept: fine scale x1.1, net 0.55, matching the norm ratio 1/1.8). Predict a change of at most +-0.05 CER, within noise; keep only if the rule says so. If it fails, net 0.5 stays.
H3 (after H2 fails: scale x0.85, net 0.425). Predict a change of +-0.05, likely not kept. If it fails, the 0.5 point is near the optimum.
H1b (if H1 not kept: x0.7 from the original, then x0.5 of that). Predict a CER fall of at least 1 point if the global-gain hypothesis is partly right. If both fail, I conclude the defect is not a global scale, nothing is shipped.
Null: if the repairs give no change beyond noise, ship the original adapter (nothing kept).


6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted-adapter/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
