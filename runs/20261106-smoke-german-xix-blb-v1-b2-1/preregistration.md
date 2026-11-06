# Pre-registration — 20261106-smoke-german-xix-blb-v1-b2-1

Filled in **before the first pass** (PROTOCOL.md §2). Empty fields are a deviation.

1. **Subject:** `models/kraken/kraken-german-xix-blb-v1.mlmodel` — tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`, file sha256 `48caacb4a742b5b5ccd2a92a005a4ca701bdf7cef7859087832b6d726f5f8df4`.
   Donors (parent/siblings) and their digests:
2. **Practice set:** `data/practice/blb-train-300.jsonl` — 300 lines, list sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`.
   Source and split side: (train side only) — selection rule: see `data/practice/blb-train-300.meta.json`.
3. **Held-out set:** identity (document ids / projects), size, sha256:
   Overlap audit result (`data/practice/overlap.json`):
4. **Budget:** 2 passes.
5. **Hypotheses** (each with predicted direction and magnitude of CER change, and the edit class):
   - H1:
   - H2:
   - H3:
6. **Controls to run:** C0 no-op, C1 parent, C2 average, C3 α-sweep, C4 retrain, C5 sham, C6 blind — which, and when.
