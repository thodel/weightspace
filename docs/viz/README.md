# Where a character is found and where it is lost — `kraken-german-xix-blb-v1`

First pieces of the visualisation epic (thodel/weightspace#3): the frame map, layer probes and
the per-frame decision view. Written 07.11.2026 from `wsp.layerview` on the practice lines
`blb-train-300`. No held-out line was read, and nothing is counted against a session budget.

```bash
.venv/bin/python -m wsp.layerview --model models/kraken/kraken-german-xix-blb-v1.mlmodel \
    --practice data/practice/blb-train-300.jsonl --out docs/viz          # about 4 min on 16 CPU threads
```

| file | in git | what |
|---|---|---|
| `kraken-german-xix-blb-v1__blb-train-300.json` | yes | folds, λ per fold, per-layer metrics, and per line the aligned characters with a decodability string per layer and probe |
| `probe-kraken-german-xix-blb-v1__blb-train-300.svg` (`-dark.svg`) | yes | the probe figure below; no manuscript image in it |
| `lines/<model>__<key>.svg` | **no** (`.gitignore`) | per-line decision views. They show the manuscript crop, so they are regenerated rather than published |

Inputs: model sha256 `48caacb4…` (matches `models/DIGESTS.md`), practice list sha256 `4a803b49…`. The output is
deterministic: two runs write byte-identical JSON.

## Method

- **Frame map.** The three 2×2 max-pools make one output frame of `O_18` cover 8 input columns.
  Every block's output (`KrakenSubject.activations`) is mean-pooled onto that grid. "pixels" is the
  8 × 120 raw pixels of a frame.
- **Forced CTC alignment.** The reference is aligned to the model's own log-posteriors by a Viterbi
  pass. This gives every reference character a run of frames, including the characters the model
  deletes. Each character is marked as emitted, replaced or deleted from the edit alignment of
  reference and best-path hypothesis. The best path equals the committed C0 reading on 39 of 40
  lines checked; the 40th, along with two other lines, had too few frames to align and was skipped.
- **Two linear probes per layer.** Both are ridge readouts of a single frame, with features standardised.
  - The *identity* probe answers which character. It is trained on character frames only and read as
    the argmax over characters.
  - The *emission* probe answers character or blank. Its λ is chosen by balanced accuracy, because
    four frames in five are blank.
  - A character counts as *decodable* when the probe gives it on at least one frame of its run.
- **Cross-fitting.** There are 5 folds by document. Probes for a test fold are trained on three
  other folds, and λ is chosen on the fifth. Every one of the 297 lines is therefore scored by probes
  that never saw its document.
- **A single probe would mislead.** With one probe over all 297 classes and λ chosen on frame
  accuracy, every conv layer predicted "blank" on every frame (frame accuracy 0.81, the blank share).
  The two-probe split is the fix. For the same reason the identity panel leaves spaces out: the space
  is the commonest class, so a weak probe "finds" it everywhere (0.99 already from pixels).

## What the probes say

| layer | pixels | C_0 | C_3 | C_6 | C_9 | L_12 | L_14 | L_16 | O_18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity, letters emitted correctly (10,004) | 0.06 | 0.10 | 0.14 | 0.24 | 0.36 | 0.59 | 0.82 | **0.94** | 0.92 |
| identity, letters deleted (284) | 0.01 | 0.03 | 0.04 | 0.06 | 0.06 | 0.11 | 0.12 | 0.18 | 0.16 |
| emission, letters emitted correctly | 0.10 | 0.22 | 0.33 | 0.42 | 0.52 | 0.70 | 0.90 | **0.99** | 0.98 |
| emission, letters replaced (307) | 0.10 | 0.24 | 0.37 | 0.46 | 0.43 | 0.58 | 0.80 | 0.91 | 0.90 |
| emission, spaces emitted (2,090) | 0.03 | 0.12 | 0.37 | 0.58 | 0.64 | **0.86** | 0.96 | 0.98 | 0.98 |
| emission, spaces deleted (77) | 0.01 | 0.16 | 0.19 | 0.25 | 0.21 | 0.31 | 0.44 | 0.47 | 0.43 |

Chance for identity (always guessing the commonest letter) is 0.14.

1. **Which letter it is gets decided in the BiLSTMs.** From a single frame of the conv stack, a linear
   readout gets about a third of the correctly read letters (0.36 at `C_9`). Each BiLSTM adds a large
   step (0.59, 0.82, 0.94), and `O_18` adds nothing. The probe sees one frame, so this measures what
   is present at that position and layer, not what the conv stack could deliver with context. The
   order still matches the α-sweep: the recurrent stack holds what is localisable.
2. **Whether to emit is also decided in the BiLSTMs, but word gaps show earlier.** For spaces, emission
   is already 0.58 at `C_6` and 0.86 at `L_12`; letters reach that level only at `L_14`.
3. **Most deleted letters are never read at all.** Their identity stays near chance at every layer
   (at most 0.18). 259 of the 361 deletions (72 %) are decodable at no layer. These characters are not
   lost inside the network: they never form. Unreadable ink, or a reference that does not match the
   crop, is the likelier cause, and no weight edit recovers what no layer has.
4. **Deleted spaces are the exception, and they are lost at the very end.** For 31 of the 77 deleted
   spaces (40 %), the `O_18` probes still have both identity and emission, yet the model's own argmax
   emits blank. The `O_18` probe is a linear function of the output logits. "Decodable at `O_18` but
   not emitted" therefore means that a different affine read-out of the same logits, i.e. a change to
   the output layer's space row or bias, would recover these spaces. Across the 77, emission sits
   around 0.44–0.47 from `L_14` on, against 0.96–0.98 for kept spaces. Together with the α-sweep
   (half of `L_16`'s fine-tune move costs 2.3 points, mostly as space deletions), this places the
   spacing decision in `L_16` → `O_18`.
5. **For replaced letters the network knows that something is there, not what.** Their emission is
   normal (0.91 at `L_16`), but their identity stays at 0.19.

## Caveats

- `blb-train-300` holds the child's own training lines. Deletions here are deletions on lines it
  was trained on.
- The forced alignment uses the child's own posteriors. A deleted character is placed where the model
  is least sure it is blank, which is the most favourable spot for the probes.
- The probes are ridge regressions to one-hot targets, not calibrated classifiers. "Decodable" is an
  argmax criterion. The `O_18` probe re-fits its own threshold, which is the point of finding 4 and
  also its limit: the re-fit is balanced and the model's own read-out is not.
- One model, one practice set. The Bern family (`bern-hertwig-train-300`) runs with the same command.

## Candidate hypotheses (not preregistered)

- **H-space:** raise the bias of the space row of `O_18`. Predicted: part of the 31 spaces lost at
  `O_18` come back, at the cost of inserted spaces. Phase 1's blank-bias edits traded deletions for
  insertions one to one, so the session has to show the trade is better for this single row.
  Measuring the trade-off curve on practice first is cheap.
- **H-L16:** the remaining deleted spaces are already weak at `L_14`/`L_16`. The per-line α overlay
  (epic sub-task 4) on these lines would show whether moving `L_16` further along the fine-tune
  (α > 1) strengthens their emission. The corpus-level α = 1.5 was a null.
- **Not a target:** letter deletions (72 % are readable at no layer) and letter substitutions. They
  point to the data or to the conv/LSTM features as a whole, not to a block an edit could address.

## Epic status (thodel/weightspace#3)

| sub-task | state |
|---|---|
| activation capture | done: `KrakenSubject.activations` (from PR #10) plus the frame map here |
| per-frame decision view | done, as static SVG per line: image, conv channel energy, BiLSTM ‖h‖ forward and reverse, CTC posteriors, reference against hypothesis, probes per layer |
| layer probes | done (this page) |
| ablation and α overlay per line | open |
| output-layer geometry | partly done: `protocol/docs/figures/output-rows-*` |
| confusion-to-frame index | open (the JSON already holds runs and status per character) |
| HTML page | open |
| readings into `docs/READINGS.md` | candidates above, not yet in READINGS |
