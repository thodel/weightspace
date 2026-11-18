# Pre-registration — 20261118-trocr-noop-smoke-gpu-b2-1

Filled in **before the first pass** (PROTOCOL.md §2). Empty fields are a deviation.

1. **Subject:** `/home/tobias/atr-cache/trocr/dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2` — tensor digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`, file sha256 `054474de1f621f61d1284ee4f274c58def0bb887b1a91694081960fb05dbaf3c`.
   Donors (parent/siblings) and their digests:
2. **Practice set:** `/tmp/claude-1007/-home-tobias-Repo-weightspace--claude-worktrees-trocr-stage0-9ababb/a9c70032-5b44-4081-b223-c2999dd94ae9/scratchpad/tiny8.jsonl` — 8 lines, list sha256 `4ff8bbaa0efb04a9e7397895418123e935850b44621d58dbe99067a08150cfa9`.
   Source and split side: (train side only) — selection rule: see `data/practice/tiny8.meta.json`.
3. **Held-out set:** identity (document ids / projects), size, sha256:
   Overlap audit result (`data/practice/overlap.json`):
4. **Budget:** 2 passes.
5. **Hypotheses** (each with predicted direction and magnitude of CER change, and the edit class):
   - H1:
   - H2:
   - H3:
6. **Controls to run:** C0 no-op, C1 parent, C2 average, C3 α-sweep, C4 retrain, C5 sham, C6 blind — which, and when.
