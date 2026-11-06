# weightspace

Weight-space programming for the in-house kraken HTR models: read the weights,
diagnose, patch, measure — and compare the result with ordinary merging and
retraining. Private repo. Plan in [PLAN.md](PLAN.md), model inventory in
[docs/MODELS.md](docs/MODELS.md).

Method source: R. Tanaka, "Weight-space programming" (blog, 2 Nov 2026),
https://rikiyat.github.io/blog/weight-space-programming/ , logs and harness at
https://github.com/RikiyaT/lm_understand-logs .

Status 06.11.2026: plan, design, protocol and the engine-free core of the harness with tests (`python3 -m pytest tests`); nothing run on asteraix yet.

Docs: [PLAN.md](PLAN.md) · [docs/DESIGN.md](docs/DESIGN.md) · [docs/PROTOCOL.md](docs/PROTOCOL.md) · [docs/ASTERAIX_RUNBOOK.md](docs/ASTERAIX_RUNBOOK.md) · [docs/MODELS.md](docs/MODELS.md) · [docs/PAPER.md](docs/PAPER.md)

Layout:

```
wsp/        metrics, ops, diff, align, session (budget + log), subject interface;
            kraken_subject (CoreML, asteraix), lora_subject (peft adapters, phase 5)
tests/      engine-free tests; tests/asteraix/ needs kraken 7.0.2 and a model
runs/       one folder per WSP session: hypotheses, edits, budget, scores
docs/       MODELS.md (inventory), READINGS.md (per-model readings), RESULTS.md
models/     local copies of weights (git-ignored)
data/       practice / held-out line sets (git-ignored)
```
