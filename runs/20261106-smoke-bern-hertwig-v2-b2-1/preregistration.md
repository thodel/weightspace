# Pre-registration — 20261106-smoke-bern-hertwig-v2-b2-1

Filled in **before the first pass** (PROTOCOL.md §2). Empty fields are a deviation.

1. **Subject:** `models/kraken/kraken-bern-hertwig-v2.mlmodel` — tensor digest `f216564e4ddcf8f080556683626798cc2574bb90ee0a196b499a5d1ce9831572`, file sha256 `479392ab7899b831b3c3872dc0f178a894495757b2df2047f2b695c4a5dcc8f3`.
   Donors (parent/siblings) and their digests:
2. **Practice set:** `data/practice/bern-hertwig-train-300.jsonl` — 300 lines, list sha256 `bd4e5a8a4957ac59d2c9126adae1c2006d860543f2b4a9fb94cbc5953a7fe25e`.
   Source and split side: (train side only) — selection rule: see `data/practice/bern-hertwig-train-300.meta.json`.
3. **Held-out set:** identity (document ids / projects), size, sha256:
   Overlap audit result (`data/practice/overlap.json`):
4. **Budget:** 2 passes.
5. **Hypotheses** (each with predicted direction and magnitude of CER change, and the edit class):
   - H1:
   - H2:
   - H3:
6. **Controls to run:** C0 no-op, C1 parent, C2 average, C3 α-sweep, C4 retrain, C5 sham, C6 blind — which, and when.
