Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API).
Reasoning effort: provider default. Tools: `read_file`, `write_file` (automatic `py_compile`), `finish_round`. No evaluation passes.

**Hypothesis:** The readings do not identify a defect precisely enough to repair blind. At C0, space deletions outnumber space insertions (103 vs 22), but the length ratio is 0.971 and deletions span several common characters. This does not distinguish an output-bias error from omissions on difficult lines. The blank bias moved only +0.113 from the parent, and there is no dominant character substitution pointing to swapped rows.

The clearest hidden-layer outlier by relative change is `C_3.co.bias` (1.0871), but the entire child vector has L2 norm only 0.10 and the corresponding kernel retains cosine 0.986 with the parent. Relative change on this small vector is not sufficient evidence for a transplant. Recurrent weight changes are broadly similar across directions; they do not single out a damaged block.

**Edit:** None. Freeze the child unchanged rather than guess a bias correction or undo handwriting-specific adaptation. **Predicted effect:** zero change in practice CER (7.26%, Δ 0 percentage points) under identical evaluation.
