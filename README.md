# weightspace

Weight-space programming for the in-house kraken HTR models: read the weights,
diagnose, patch, measure — and compare the result with ordinary merging and
retraining. Private repo. Plan in [PLAN.md](PLAN.md), model inventory in
[docs/MODELS.md](docs/MODELS.md).

Method source: R. Takehi, J. Teoh, A. Zhang, O. Khattab, "Sparks of Weight-Space Programming" (blog, 2 Nov 2026),
https://rikiyat.github.io/blog/weight-space-programming/ , logs and harness at
https://github.com/RikiyaT/lm_understand-logs .

Status 14.11.2026: see [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md). The kraken and adapter tracks are done and merged. One budgeted session, the Fable 5.1 rerun of kraken session 2 ([docs/KRAKEN_S2_FABLE.md](docs/KRAKEN_S2_FABLE.md)), gave a held-out gain whose page interval excludes 0. It matches a short retrain on corrected data and does not beat it; the same retrain on the uncorrected pairs gains nothing. One block-exchange candidate (`latin_incunabula` v3) qualifies (registered, disabled). No pull request is open. A TrOCR track has passed stages 0 and 1 ([docs/trocr/STAGE0.md](docs/trocr/STAGE0.md), [docs/TROCR_READINGS.md](docs/TROCR_READINGS.md)). The activation atlas has levels L0–L3 ([docs/ATLAS_REPORT.md](docs/ATLAS_REPORT.md)); Gate 2 fails for both models, so it stays descriptive.

Public protocol (MkDocs, for GitHub Pages later): [protocol/](protocol/README.md).

Current state: [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) (tracks, findings, open items) · article: [paper/](paper/README.md), [paper/STATUS.md](paper/STATUS.md).

Docs: [PLAN.md](PLAN.md) · [docs/DESIGN.md](docs/DESIGN.md) · [docs/PROTOCOL.md](docs/PROTOCOL.md) · [docs/ASTERAIX_RUNBOOK.md](docs/ASTERAIX_RUNBOOK.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/PAPER.md](docs/PAPER.md) · [docs/RELATED_WORK.md](docs/RELATED_WORK.md) (how prior work assessed text handling) · [docs/SEARCH_LOG.md](docs/SEARCH_LOG.md) (how the literature was searched) · [docs/DEPLOY_ASTERAIX.md](docs/DEPLOY_ASTERAIX.md) (bootstrap commands) · [docs/BRIEF_PHASE0.md](docs/BRIEF_PHASE0.md) (what to do on asteraix first) · [docs/RESULTS.md](docs/RESULTS.md) (R1-R14) · [docs/CONTROLS_REPORT.md](docs/CONTROLS_REPORT.md) · [docs/SEED_NULL.md](docs/SEED_NULL.md) · [docs/ZOO_SURVEY.md](docs/ZOO_SURVEY.md) · [docs/MODEL_FAMILY.md](docs/MODEL_FAMILY.md) · [docs/C4_RETRAIN.md](docs/C4_RETRAIN.md) · [docs/C4_UNCORRECTED.md](docs/C4_UNCORRECTED.md) · [docs/ADAPTER_REPORT.md](docs/ADAPTER_REPORT.md) · [docs/PHASE5_REPORT.md](docs/PHASE5_REPORT.md) · [docs/epics/](docs/epics/EPIC_LLM_PLAYBOOK.md) · [docs/MODELS_TROCR.md](docs/MODELS_TROCR.md) · [docs/epics/EPIC_TROCR_TRACK.md](docs/epics/EPIC_TROCR_TRACK.md)

Layout:

```
wsp/        metrics, ops, diff, align, session (budget + log), subject interface;
            kraken_subject (CoreML, asteraix), lora_subject (peft adapters, phase 5)
tests/      engine-free tests; tests/asteraix/ needs kraken 7.0.2 and a model
deploy/     asteraix_bootstrap.sh — idempotent setup on asteraix
runs/       one folder per WSP session: hypotheses, edits, budget, scores
docs/       MODELS.md (inventory), READINGS.md (per-model readings), RESULTS.md
models/     local copies of weights (git-ignored)
data/       practice / held-out line sets (git-ignored)
```
