# weightspace

Weight-space programming for the in-house kraken HTR models: read the weights,
diagnose, patch, measure — and compare the result with ordinary merging and
retraining. Private repo. Plan in [PLAN.md](PLAN.md), model inventory in
[docs/MODELS.md](docs/MODELS.md).

Method source: R. Takehi, J. Teoh, A. Zhang, O. Khattab, "Sparks of Weight-Space Programming" (blog, 2 Nov 2026),
https://rikiyat.github.io/blog/weight-space-programming/ , logs and harness at
https://github.com/RikiyaT/lm_understand-logs .

Status 06.11.2026 (evening): phase 0 executed on asteraix — harness verified against kraken 7.0.2, practice sets built and audited, C0 baselines, the lossless `Cr255 → Cr1` rewrite, the session runner with a no-op smoke session, and the first readings. Report: [docs/PHASE0_REPORT.md](docs/PHASE0_REPORT.md). One data finding changes target 1: the 25.10 re-export of the Hertwig dataset pairs 44 of 50 training pages with the wrong image.

Public protocol (MkDocs, for GitHub Pages later): [protocol/](protocol/README.md).

Docs: [PLAN.md](PLAN.md) · [docs/DESIGN.md](docs/DESIGN.md) · [docs/PROTOCOL.md](docs/PROTOCOL.md) · [docs/ASTERAIX_RUNBOOK.md](docs/ASTERAIX_RUNBOOK.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/PAPER.md](docs/PAPER.md) · [docs/RELATED_WORK.md](docs/RELATED_WORK.md) (how prior work assessed text handling) · [docs/SEARCH_LOG.md](docs/SEARCH_LOG.md) (how the literature was searched) · [docs/DEPLOY_ASTERAIX.md](docs/DEPLOY_ASTERAIX.md) (bootstrap commands) · [docs/BRIEF_PHASE0.md](docs/BRIEF_PHASE0.md) (what to do on asteraix first)

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
