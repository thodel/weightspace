Model: claude-haiku-5-5

# Preregistration, kraken-german-xix-blb-v1 (session h1)

Written from the pack only (`describe.md`, `diff_parent_to_subject.md`, `c0_practice.md`) and the wsp API files, before any pass. No model was loaded and no pass was run.

## Settings
- Subject: `kraken-german-xix-blb-v1` (CTC, 4.11 M parameters, output layer `O_18.lin`, row 0 = blank, 297 rows).
- Donor: `kraken-german-xix-v1` (parent). The pack says it never saw BLB data and `seen_by` is empty for every practice line, so the donor rule mask should be all lines. The program still checks the mask and skips the donor edit if the mask is empty.
- Practice set: blb-val-300 (300 handwritten lines from 7 unseen objects). C0 baseline CER 0.1383, WER 0.3625, length ratio 0.970, 2 collapsed lines. The parent scores CER 0.3266 on the same lines.
- C0 error profile: the errors are mostly deletions (reference → nothing: space 78, e 44, i 29, n 26, r 23, t 21, s 21). Insertions are rarer (nothing → space 30). The weakest output rows are the blank (norm 4.64, bias +1.23) and the letters e, r, s, t, i, n, a (norm 7.6 to 8.3, against a mean of about 14.8). The space row has bias -7.85.
- Budget: 10 passes. The program uses at most 6 (baseline plus five edits, one of them conditional).
- Keep rule: the session default `min_effect = 0.0`, so an edit is kept only if its paired 95 % interval of CER(after) minus CER(last kept) lies entirely below 0. I assume the harness uses this default (the pack does not say otherwise).
- Expectation about the rule: with 300 lines I expect the paired interval half-width to be roughly 0.003 to 0.005 CER. So an edit with a predicted change smaller than about 0.008 will probably be refused by the interval rule even if the point estimate is negative. This is a prediction in its own right: edits will be refused more often than they are kept.

## Hypotheses (in run order)

**H1. Lower the blank bias.** Edit class: scalar output bias (`add` of -0.3 on `O_18.lin.bias[0]`).
- Prediction: CER falls by about 0.010 (range 0.003 to 0.03). Length ratio rises from 0.970 toward 0.98 to 0.99. Deletions fall more than insertions rise.
- If it fails (refused, or CER rises): the blank is not what suppresses the letters. The deletions come from the weak letter rows, not from blank competition.

**H1b. Lower the blank bias further (only if H1 is kept).** Edit class: scalar output bias, another -0.3.
- Prediction: a further CER fall of about 0.005, with diminishing returns. Insertions start to rise, so the gain is smaller than H1's.
- If it fails: the first step is near the optimum for the blank, and the program keeps the H1 state.

**H2. Raise the space row bias.** Edit class: scalar output bias (`add` of +1.5 on the space row, bias -7.85 to -6.35).
- Prediction: CER falls by about 0.004 (range 0.001 to 0.01). Space deletions (78) fall, and space insertions (30) rise slightly. Word error rate falls more than CER.
- If it fails: a bias step of this size cannot move a row that sits at -7.85, so spaces are not recoverable through the bias alone.

**H3. Enlarge the weakest letter rows.** Edit class: row-wise weight edit (`add` of 0.1 times the row, i.e. a ×1.1 scale, on the rows of e, i, n, r, t, s in `O_18.lin.weight`).
- Prediction: CER falls by about 0.003 (range -0.01 to +0.003). The sign is uncertain. Deletions of these letters fall. Substitutions among them, such as r → n and r → e, may rise.
- If it fails (refused or CER rises): the row norm is not the cause of the deletions. Their cause is probably the hidden states and the recurrent layers.

**H4. Pull the output layer toward the parent (negative control, donor edit).** Edit class: interpolation with a donor, α = 0.05, applied to `O_18.lin.weight` and `O_18.lin.bias`, rows aligned by character with `codec_row_map`. Characters the parent lacks keep the child's rows. The donor rule mask is `s.unseen_by("kraken-german-xix-v1")`.
- Prediction: CER rises by 0.000 to +0.01, since the parent scores 0.327 on these lines and the child 0.138. The keep rule refuses the edit.
- If it is kept: that would be surprising. It would mean the parent's output rows help on lines it never trained on, and the child's output layer would then be overfitted to BLB. I would report this as a finding and not as an improvement.

## Controls and what I will report
- The keep rule decides every edit, and the session is frozen at the end. The held-out score is not read by the programmer.
- A kept edit is not an improvement on the held-out set. Only the frozen state and its `summary.json` are the result.
