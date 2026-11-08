# Pre-registration — 20261107-planted-adapter-p3-alpha2-c-b4-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (c). Written before the first pass; the program is `runs/planted-adapter/arms/program_c.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 4 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

(the rules are in the header of `runs/planted-adapter/arms/program_c.py`)

6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** none: a fixed hand-written list of repairs (runs/planted-adapter/arms/program_c.py), written by the session's author (Claude Sonnet 5.5) before any pass, identical for every adapter.
