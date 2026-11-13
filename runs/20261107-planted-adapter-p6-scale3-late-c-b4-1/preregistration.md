# Pre-registration — 20261107-planted-adapter-p6-scale3-late-c-b4-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (c). Written before the first pass; the program is `runs/planted-adapter/arms/program_c.py`, copied unchanged.

1. **Subject:** `p6-scale3-late`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "layers 24-31 mlp.down_proj (B x3)", "lora_alpha": 128, "digest": "200a131372d6e967022e2fdb4ee0ff2ab0665c8f359cc15064895795c6f26fc2", "added": "after p1/p2/p4 were null"}); tensor digest `c337deaf5979f87933f9d285e24b50acd9b6d9b9531f1c7efab886938813e747`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 4 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

(the rules are in the header of `runs/planted-adapter/arms/program_c.py`)

6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** none: a fixed hand-written list of repairs (runs/planted-adapter/arms/program_c.py), written by the session's author (Claude Sonnet 5.5) before any pass, identical for every adapter.
