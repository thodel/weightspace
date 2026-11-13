# Hypotheses — planted-f1 / s01 / task (b), budget 0 (blind programmer)

Model id: claude-fable-5-1 (Claude Fable 5.1). Reasoning effort: low (as configured by the caller, not raised).
Tools used: Read / Bash (cat, ls, sed, grep) on the three pack files, the brief, `wsp/run.py` and `wsp/session.py`;
`py_compile` on `program.py`. No model loaded, no pass run, nothing else in the repository read (the sibling `a/` folder untouched).

## Readings, in order of weight

1. **`C_3.co.weight` [32,32,3,13] and `C_3.co.bias` [32] are exactly zero** (L2 0.00, rms 0, abs max 0; diff table: rel L2
   1.0000 with cosine `nan`, i.e. the subject's tensor has no direction). Every other tensor has an ordinary norm for its shape.
2. **The practice reading is a dead model**: CER 1.0000, WER 1.0000, 300/300 lines collapsed, length ratio 0.000, every one of
   the 12,867 reference characters is a deletion (`x → ∅`), no insertion and no substitution anywhere.
3. Output layer: blank row norm 4.64, bias +1.23; frequent characters (`e r s t i n a`) have the weakest rows (7.6–8.3) and biases
   near zero; rare diacritic letters the strongest (≈23). This is the normal shape of a CTC output layer for a frequency-skewed
   codec. Space has bias −7.85 (the lowest of all 297) with row norm 18.35, which is unusual for the most frequent symbol, but
   space is the one class whose evidence is "no ink", and I know of no healthy reference value for this lineage (the donor has
   no shared space row). Not actionable.
4. `L_16.layer.weight_ih_l0` abs max 4.70 against 1.3–2.2 for the five other LSTM input matrices; its rms (0.242) is within
   10 % of theirs. A single large weight in an LSTM is not evidence of damage. Not actionable.

Readings 1 and 2 explain each other completely. `C_3` is `Cr3,13,32` → `Mp2,2`: with weight and bias both zero its output is
ReLU(0) = 0 for every pixel of every line. Everything downstream then sees a constant input: C_6 outputs ReLU(bias) constants,
C_9 likewise, the three BiLSTMs settle to a fixed state over a constant sequence, the output layer emits one fixed logit
vector at every time step, and the blank (bias +1.23, the only row favoured a priori) wins at every step. Greedy CTC decoding
of an all-blank sequence is the empty string — hence exactly the observed reading: nothing but deletions, length ratio 0,
all lines collapsed. I am sure about the diagnosis.

## H1 — the only defect is a zeroed second convolution block (`C_3`)

- **Edit that would fix it:** `transplant` of `C_3.co.weight` and `C_3.co.bias` from the model's own parent
  (`kraken-german-xix-v1`). Predicted effect of *that* edit: CER from 1.00 back to the parent's level (large improvement,
  hundreds of edits per 1 % of CER), collapsed lines 300 → ≈0.
- **Why it is not applied:** the parent is not available. The only donor permitted, `kraken-bern-hertwig-v2`, is from a
  different lineage: its `C_0` has cosine 0.09 with ours and its `C_6` cosine 0.06, so its `C_3` was trained to map *its*
  `C_0` features onto *its* `C_6` input basis — neither of which this model shares. Transplanting it does not give C_6 the
  channels it was trained on; it feeds the intact LSTM stack and output layer a structured but foreign feature stream.
  Prediction for that edit: output changes from empty strings to garbage strings of roughly the time-step count (≈ width/8,
  often several times the reference length); CER likely ≥ 1.0 and plausibly 1.5–3 because every surplus character is an
  insertion, where the dead model's CER is exactly 1.0 (deletions only). Same reasoning for an identity/pass-through `C_3`
  (centre tap = δ_ij): C_6 would receive C_0 features instead of C_3 features, again a basis it never saw.
- **Conclusion:** no edit with the donor at hand is one I am "fairly sure" improves the model; the plausible outcome is a
  worse CER than the dead baseline. Under the budget-0 rule (nobody checks before scoring) I ship **no edit**.

## H2 — space row suppressed (bias −7.85) — NOT applied
- Edit would be `add` to `O_18.lin.bias[cm[' ']]` of ≈ +5. Predicted effect on its own: none on the practice set (the model is
  dead before the output layer; at most the constant per-step argmax could flip from blank to space, CER unchanged ≈ 1.0).
- Confidence too low that it is a defect at all (see reading 3), and harmful if it is not. Not applied.

## H3 — `L_16.weight_ih_l0` outlier — NOT applied
- Would be a `set_rows`/clip of the offending entries. One value of 4.7 in an 800×400 matrix with normal rms is not evidence.
  Not applied.

## What the program does
`program.py` opens the session, records the diagnosis and the reasons for applying nothing as `s.note(...)` entries, applies
**no edit**, and calls `s.freeze()`. The frozen model is byte-for-byte the subject. Predicted practice CER after the program:
1.0000 (unchanged). What is needed to repair this model is a `transplant` of the `C_3` block from `kraken-german-xix-v1`, or
retraining; neither is possible in this session.
