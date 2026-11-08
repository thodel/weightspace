# Pre-registration — 20261107-planted-k3-swap-rows-e-n-c-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (c). Written before the first pass; the program is `runs/planted/arms/program_c.py`, copied unchanged.

1. **Subject:** `k3-swap-rows-e-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output rows of `e` and `n` swapped (weights and bias); `runs/planted/plant.json`) — tensor digest `e04ecfb521c6e879805ac6017fbecdaa4545f1e7e315de4804ea722f62b5ba9c`, file sha256 `c0bc09e5c852501c31d1fd20b48a9a2495071d13fa61edaf919573fb89b580e3`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

(the rules are in the header of `runs/planted/arms/program_c.py`)

6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** none: a fixed hand-written list of repairs (runs/planted/arms/program_c.py), written by the session's author (Claude Sonnet 5.5) before any pass, identical for every model.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
