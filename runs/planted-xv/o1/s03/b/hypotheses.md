Model: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API).
Settings: provider-default reasoning; tools used: read_file, write_file (automatic py_compile), finish_round. Zero evaluation passes.

**H1 — excess CTC blank bias.** The blank bias is +3.232, up 2.113 from the parent, whereas the blank weight row changed by about as much as an ordinary shared row. The decoded text is 5.9% short and all ten leading confusions are deletions, across spaces and several common letters rather than one character pair. Subtract 2.0 from `O_18.lin.bias[0]` only: this puts the blank intercept near its parent value without replacing the child's learned visual classifier. I predict fewer deletions (especially of spaces), a length ratio closer to one, and a net CER decrease of roughly 1–3 percentage points from 7.89%; some extra insertions are possible. The exact excess cannot be inferred from the bias difference alone.

No donor transplants or other edits: the relatively large *fractional* change in the small C_3 bias is not by itself evidence of a damaged feature channel, and the practice confusions do not implicate a particular nonblank row.
