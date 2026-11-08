# Pre-registration — 20261107-val-bern-combined-v1-replay-b10-1

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `runs/validation/arms/program_combined_replay.py` copied unchanged.

1. **Subject:** `kraken-bern-combined-v1.mlmodel` — tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`. Donors: `kraken-bern-hertwig-v2` (trained on all 6 Hertwig val pages: 212 lines seen) and `kraken-bern-schlussredaktion-v1` (trained on 8 of the 11 Schlussredaktion val pages: 216 lines seen); donor rule applied through `rule_mask`.
2. **Practice set:** `data/practice/bern-val.jsonl`, 513 lines, sha256 `30d2b536deef178ab36d07e85180fa003f45fc015e657cda79b3e6b06031dbf2`. Source: the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1 (pages_val.lst; partition 0.9, page level), 6 Hertwig pages from the 22.10 export (correct images), 11 Schlussredaktion pages. **Not training data of the subject** (audit `data/practice/overlap-bern-val.json`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines, sha256 `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

(the 8 hypotheses H1–H8 of the original preregistration `runs/20261106-bern-combined-v1-b10-1/preregistration.md` / the header of `runs/validation/arms/program_combined_replay.py`)

6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) wrote the original program on 06.11.2026; replayed here unchanged (bern: only the donor-rule masks added).
