# kraken-weightspace

Weight-space programming for the in-house kraken HTR models: read the weights,
diagnose, patch, measure — and compare the result with ordinary merging and
retraining. Private repo. Plan in [PLAN.md](PLAN.md), model inventory in
[docs/MODELS.md](docs/MODELS.md).

Method source: R. Tanaka, "Weight-space programming" (blog, 2 Nov 2026),
https://rikiyat.github.io/blog/weight-space-programming/ , logs and harness at
https://github.com/RikiyaT/lm_understand-logs .

Layout (planned):

```
wsp/        loader (.mlmodel <-> torch), eval, diff, edit log, sandbox runner
runs/       one folder per WSP session: hypotheses, edits, budget, scores
docs/       MODELS.md (inventory), READINGS.md (per-model readings), RESULTS.md
models/     local copies of weights (git-ignored)
data/       practice / held-out line sets (git-ignored)
```
