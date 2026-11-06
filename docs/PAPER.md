# Paper notes

Working title: *Weight-space programming for handwritten text recognition:
reading and patching kraken CTC models and QLoRA adapters without gradients.*

## Research questions

1. Can an LLM, reading only the weights and a budget of forward passes,
   identify defects in small HTR recognisers (CTC collapse, unlearned classes,
   damaged layers) and repair them, as Takehi et al. report for MLPs and small LMs?
2. How do such edits compare with weight averaging, delta scaling and a short
   retrain on the same held-out sets?
3. Does the result transfer from a 4 M-parameter CNN-BiLSTM-CTC model to
   rank-16 QLoRA adapters of 4–7 B vision-language page models?
4. What does the budget buy: blind (0 passes) vs 10 / 25 / 50?

## Contributions (intended)

- An engine-agnostic harness (subject / edit / session / budget / log) with
  a pre-registration protocol and published sparse diffs.
- The first application of weight-space programming to CTC recognisers and to
  LoRA adapters, with negative results reported alongside gains.
- A family-level reading of fine-tune lineages (parent → children) paired
  with behavioural confusions, and merging baselines for HTR models.

## Related work to position against (to verify and cite properly)

- Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights" (blog, 2 Nov 2026); Karpathy, Software 2.0.
- Model merging: task arithmetic (Ilharco et al. 2023), TIES-merging (Yadav
  et al. 2023), DARE (Yu et al. 2024), model soups (Wortsman et al. 2022),
  evolutionary model merge (Akiba et al. 2024).
- Model editing without training: ROME / MEMIT (Meng et al. 2022/2023),
  activation / direction ablation (Arditi et al. 2024, refusal direction).
- Mechanistic reading of weights: induction heads (Elhage et al. 2021),
  weight-space probing of CNNs.
- HTR: kraken (Kiessling), CATMuS (Clérice et al.), kraken+ (Ströbel,
  dissertation), CTC degeneration / collapse in HTR.

## Figures and tables (planned)

1. Harness diagram (subject / session / sandbox / held-out boundary).
2. Lineage diff heatmap per family, next to the confusion deltas.
3. Per-session trajectory: practice CER per pass, kept/reverted edits.
4. Main table: subject × {C0, C1, C2, C3, C4, C5, C6, WSP b10/b25/b50} on
   held-out, with bootstrap intervals and collapsed-line counts.
5. The adapter case: singular directions of ΔW per module, the direction
   removed, truncation counts before and after.

## Writing rules for this repo

Every number in the text links to a `runs/<session>/summary.json`; every
figure has a script under `scripts/figures/`; the protocol's deviations list
is reproduced in the appendix.
