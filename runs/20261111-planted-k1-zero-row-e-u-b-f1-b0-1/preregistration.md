# Pre-registration — 20261111-planted-k1-zero-row-e-u-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k1-zero-row-e-u-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s08/b/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — s08, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort, no extended tools; read only the three pack files,
the brief, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/align.py`, `wsp/subject.py` (for `apply`).
No model loaded, no pass run; `py_compile` only.

## What the readings show

1. `describe.md`: the output row of `e` in `O_18.lin.weight` has norm **0.00**. Every other row has norm > 4.6 (the blank) and the
   frequent letters sit at 7.7–8.3 (`r` 7.80, `s` 8.05, `t` 8.08, `i` 8.17, `n` 8.33, `a` 8.33). A healthy `e` row would be in that
   band. An exactly zero row of the most frequent letter of German is not a training outcome: the row was zeroed.
2. `c0_practice.md`: CER 0.1623 with 2088 edits, of which `e → ∅` 910, `e → a` 131, `e → i` 90, `e → o` 52, `e → r` 36, `e → c` 35,
   `e → u` 32: about 1,290 of the 2,088 edits (62 %) are a lost `e`. Length ratio 0.900: the output is 10 % short, i.e. the lost `e`s are
   mostly deletions. The remaining confusions (` `→∅ 103, `n`→∅ 31, `r`→∅ 25) look like the ordinary error floor of a Kurrent model.
3. `e` is not listed among the bias extremes, so its bias is between roughly −7 and +0.28 (`1` +0.28 is the 5th-highest listed). With a zero
   weight row the `e` logit is that constant at every frame. That ~30 % of the `e`s are still read means the constant is high enough to win
   where every competitor (incl. the blank) is pushed below it: the bias is probably near 0, not strongly negative.
4. `diff_donor_to_subject.md`: cosines of all conv/LSTM weights against the donor are ≈ 0 (|cos| < 0.1 everywhere, LSTM weights < 0.01);
   aligned output rows have cos ≈ 0 and rel L2 ≈ 1.6. The donor shares the architecture but none of the feature space. Nothing else
   stands out: all other blocks are of ordinary scale, `bias_ih` zero by construction, no other weak or exploded row.

**Diagnosis:** one defect, `O_18.lin.weight[cm['e'], :] == 0` (output row of `e` zeroed). Everything upstream looks intact.

## Repairs considered

| # | hypothesis / edit | edit class | predicted effect | decision |
|---|---|---|---|---|
| H1 | Transplant the donor's `e` row (and bias) into the subject | transplant (row, donor aligned by character) | The donor's row lives in the donor's LSTM feature space (cos ≈ 0 against the subject for every LSTM/conv tensor and every aligned output row), so in the subject it is a random 400-vector of norm ~16: `e` would fire at arbitrary frames. CER **up**, probably far above 0.16. | **rejected** |
| H2 | Reconstruct the `e` row from the subject's own rows (mean of `a,i,o,c,u,r`, the letters that fire where `e` is lost), plus a bias boost | set_rows (synthetic) | Mechanism plausible (ties `e` to "round-letter features active") but the margin that would make `e` win at true-`e` frames without flipping `a`/`i`/`o` centres is a one-parameter guess with no pass to set it. Sign of the effect unknown. | **rejected** (not fairly sure) |
| H3 | Raise the `e` bias so that the constant `e` logit sits just above the blank's bias (+1.23): set `O_18.lin.bias[cm['e']] = +1.5` (only if currently lower) | set_rows on the bias (1 scalar) | Addresses the 910 `e → ∅` directly: at a true-`e` centre every competitor is suppressed and the blank's logit is its bias +1.23 plus a (trained-negative) weight term; an `e` constant of +1.5 wins wherever the blank's weight term is below +0.27, and also beats the weak `a`/`i`/`o` logits that currently take `e`'s place. Letter centres (true letter ≫ 1.5) and blank frames (blank ≫ 1.5) are untouched; the risk is `e` insertions at transition frames whose maximum logit is below 1.5. Prediction: CER **down by 0.02–0.05** (recover ~400–700 of ~1,290 lost `e`s, minus some insertions); the full repair (a parent row) would give ≈ −0.10, which this edit cannot reach. | **applied**, guarded |

Guard in the program: the edit is applied only if the live tensors confirm the reading (`‖W[e]‖ < 0.5` and the current `e` bias < 1.5);
otherwise the program applies nothing and freezes. The edit does not touch the weight row: with no healthy source for it, writing any
vector there is a guess, while a bias of +1.5 is a bounded, interpretable change (a 0.27 margin over the blank's resting level) whose
failure mode (some extra `e`s) is milder than the 910 deletions it targets.

## What I would conclude afterwards

If the score is better than 0.1623: the deletions were mostly the blank beating a near-zero constant, as read. If it is worse: transition
frames have maximum logits below 1.5 more often than I assumed and `e` insertions outweigh the recovered deletions; the right value was
lower, or the row needs a real source (the parent) and no bias trick helps.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s08/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF_UNRELATED.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
