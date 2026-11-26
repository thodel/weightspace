Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API). Reasoning effort: provider default.
Tools: read_file, write_file (with py_compile), finish_round. No model runs or measurement passes.

**Hypothesis:** The child's `e` and `n` output logits have exchanged labels. The two reciprocal substitutions account for 2,201 of 3,144 edits, while the output/reference length ratio is still 0.969. Both rows have cosine near 0.36 with their own parent rows; the next most changed row has cosine 0.881. This points to a transposition of learned rows rather than a general loss of letter recognition.

**Edit:** Use `set_rows` to exchange the child's `e` and `n` weight vectors and their corresponding bias entries, identified through its codec. This inverts a full logit swap without discarding the child's adaptation. Leave the blank and every other parameter alone.

**Predicted effect:** Most reciprocal substitutions disappear. Their count corresponds to 0.171 CER, so I expect roughly a 17 percentage point absolute decrease from 24.43% to around 7%, allowing for alignment changes and genuine recognition errors. The weight exchange should account for nearly all the gain; exchanging the biases completes the permutation with little incremental CER change.
