Model: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API via the operator's tool layer).
Settings: provider default reasoning; read_file (including parallel reads), write_file with automatic py_compile, finish_round. Zero evaluation passes.

**Hypothesis:** The `n` output weight row has its sign reversed. Its cosine with the aligned parent row is −0.991 despite similar norms (8.33 vs 7.96). This accounts for both kinds of `n` error: contexts that should suppress `n` now emit it (7,667 insertions), while genuine `n` often decodes as `m` or `r`.

**Edit:** Negate only the child's `O_18.lin.weight` row indexed by its codec for `n` (`set_rows`). This restores positive alignment without discarding the child's learned adjustments. The bias delta of −0.142 is not compelling evidence of bias damage; leave all biases and other weights alone.

**Predicted effect:** The excess `n` insertions should largely disappear and the decoded/reference length ratio should approach one. I expect an absolute CER reduction of roughly 0.60–0.70 from 0.7456, not a measured result.
