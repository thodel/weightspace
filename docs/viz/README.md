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
4. **Deleted spaces look like the exception, but mostly are not.** For 31 of the 77 deleted spaces,
   the `O_18` probes still have identity and emission. This first read like a decision lost at the
   last readout. The bias trade-off below refutes that reading. Two things are behind the 31:
   - **The identity probe is trivially right on spaces** (spaces are the commonest class).
   - **Many deleted spaces are attributed, not missed.** At its best frame, the space is the top
     logit for 20 of the 31, so the model writes a space there. The edit alignment then charges the
     deletion to the reference, which spaces out punctuation (`konnte .`, `hat , von`). Of all 77
     deleted spaces:
     - 29 sit next to a deleted punctuation mark;
     - 28 sit next to a deleted letter;
     - 2 sit next to a replaced character;
     - only **18** fall between two correctly read characters, i.e. are a missing word gap.

     Emission across the 77 does stay weaker (0.44–0.47 from `L_14` on, against 0.96–0.98 for kept
     spaces). So for the deletions that are real, the spacing decision is made in the BiLSTMs, as
     the α-sweep said for `L_16`.
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

## The space-bias trade-off (07.11.2026)

`python -m wsp.bias_sweep … --probes docs/viz/kraken-german-xix-blb-v1__blb-train-300.json --verify`
→ `docs/baselines/bias-xix-blb-v1.json`, figure `bias-kraken-german-xix-blb-v1__blb-train-300.svg`.

**Method.** A bias edit only adds δ to one logit column, and kraken reads the best path. The edited
model's text is therefore re-decoded exactly from one capture of the logits, with no pass per δ.
This was checked against real edited forward passes: 300 of 300 lines identical at space +1.0 and
at blank −1.0. Blank −1.0 also reproduces phase 1's recorded practice CER (0.06435).

| lever | best ΔCER (points), 95 % interval | spaces deleted / inserted (unedited 103 / 18) | of the 31 flagged spaces emitted |
|---|---|---|---:|
| space row bias, δ +0.5 … +1.0 | +0.047 [−0.045, +0.145] | 97 / 24 at +1.0 | 5 |
| space row bias, δ +3 | +0.451 [+0.236, +0.685] | 108 / 35 | 5 |
| blank bias, δ −1.0 | −0.023 [−0.324, +0.309] | 73 / 31 | 14 |
| blank bias, δ −3 | +1.119 [+0.460, +1.851] | 36 / 78 | 20 |

**Result.**
- **H-space is refuted.** At no δ does the space row's bias lower CER. It never brings back more
  than 5 of the 31 flagged spaces. Past +2 it deletes *more* spaces, because neighbouring space frames
  lose the blank between them and merge.
- **The blank bias moves along an almost one-to-one trade.** Deleted spaces fall from 103 to 73 and
  inserted spaces rise from 18 to 31. That is phase 1's null, now with its whole curve.
- **No output-layer bias is a target on this model.** Below the output layer, see the α overlay next.

## Which block's share of the fine-tune does what: the α overlay (07.11.2026)

```bash
.venv/bin/python -m wsp.alpha_overlay --child models/kraken/kraken-german-xix-blb-v1.mlmodel \
    --parent models/kraken/kraken-german-xix-v1.mlmodel --practice data/practice/blb-train-300.jsonl \
    --out docs/viz --check docs/baselines/alpha-xix-blb-v1.json                      # about 9 min
```

**What it computes.** It repeats the C3 α-sweep, one block at `parent + α(child − parent)` with
α ∈ {0, 0.5, 1.5} and the output layer's rows paired by character. This time it keeps every line's
CTC strip and the fate of every reference character. The 16 CERs it shares with the committed
sweep (`docs/baselines/alpha-xix-blb-v1.json`) are identical. Files:
- `alpha-overlay-kraken-german-xix-blb-v1__blb-train-300.json`: per block and α, the CER and the
  edits by kind, plus per line the status string;
- `alpha-overlay-…svg`: the summary figure;
- `lines/alpha-…svg`: per-line views (git-ignored).

**Reading it.** α = 0 replaces one block of the child with its parent's, so it isolates that block's
share of the fine-tune; the rest of the network stays the child. The table counts edits against the
unedited child. The child itself deletes 354 letters and 103 spaces on these 300 lines, at
CER 6.46 %.

| block at α = 0 | ΔCER (points) | letters deleted | spaces deleted | substituted | inserted |
|---|---:|---:|---:|---:|---:|
| C_0 | +0.27 | +6 | −1 | +30 | +0 |
| C_3 | +0.45 | +3 | +0 | +47 | +8 |
| C_6 | +0.80 | +10 | +3 | +52 | +38 |
| C_9 | +0.65 | +18 | +3 | +47 | +16 |
| **L_12** | +6.37 | **+366** | +67 | +301 | +86 |
| **L_14** | +4.05 | +96 | +27 | **+342** | +56 |
| **L_16** | +5.70 | +100 | **+362** | +270 | +1 |
| O_18 | +0.68 | +37 | +38 | +26 | −14 |

What the fine-tune from `german-xix-v1` to `german-xix-blb-v1` taught, block by block:

1. **`L_12`: finding the letters.** With its parent version, the child drops 366 more letters and
   inserts 86 more characters. Its share of the move is what makes letters appear at all on the
   BLB hands.
2. **`L_14`: telling the letters apart.** It adds substitutions (+342) and relatively few deletions.
3. **`L_16`: the word gaps.** It adds 362 deleted spaces, 3.5 times the child's own count, almost
   no insertions, and comparatively few letter deletions. This confirms the phase-2 reading (half of
   `L_16`'s move costs 2.3 points, mostly spaces) and pins it down: the fine-tune taught the BLB
   spacing in the last BiLSTM.
4. **The conv stack: detail of shape.** Each conv block's share costs 0.3–0.8 points, almost all of
   it substitutions.
5. **`O_18`: small and balanced.** Its share adds deletions and removes a few insertions, which is
   the blank and rare-row shift the phase-0 readings saw.

At α = 1.5 no block improves CER. `L_16` (+0.05) trades 20 fewer substitutions for 28 more letter
deletions. The corpus null of the sweep hides an exchange, not a dead lever.

**Together with the probes.** Identity forms over `L_12` → `L_16`, emission of word gaps matures
in the BiLSTMs, and what the fine-tune changed in those blocks is, in order: letters found
(`L_12`), letters told apart (`L_14`), words separated (`L_16`). This is the answer so far to
"which part of the recognition process happens where" for this model.

Caveats as for the probes: these are the child's training lines, one model, and one block at a time,
so interactions between blocks (the α-sweep found the gain is held jointly) are not shown.

## The page: `kraken-german-xix-blb-v1.html`

Open the file from disk; it needs no server. Pick any of the 297 practice lines, filtered by text,
by deletions, by deleted spaces, or by CER ≤ 20 % (on by default, because worse lines are mostly
crop or reference defects). The line crop sits on top; each reference character is placed under
its ink by the forced alignment. Below that:
- the model's reading;
- per layer, whether the probes have each character (■ identity and emission, □ identity only);
- per block and α, the character's fate when that block alone is moved towards the parent.

Hovering a character gives its column, including the deepest layer that still has it and every
(block, α) that changes its fate.

```bash
.venv/bin/python -m wsp.vizpage --probes docs/viz/kraken-german-xix-blb-v1__blb-train-300.json \
    --overlay docs/viz/alpha-overlay-kraken-german-xix-blb-v1__blb-train-300.json \
    --practice data/practice/blb-train-300.jsonl --out docs/viz/kraken-german-xix-blb-v1.html --images
```

The page reads only the committed JSON and embeds what it needs (1.2 MB). `--images` writes the
line crops to `docs/viz/img/`. They are git-ignored, so on a fresh checkout the page works without
the image row until `--images` has run. That step needs kraken but not the model.

## Output-layer geometry (08.11.2026)

`python -m wsp.outgeom --parent <parent> --child <child> --practice <set> --out docs/viz` (no forward pass). The
output layer has one row per character and one for the CTC blank; parent and child rows are paired **by
character**. Three panels: the rows of both models in one 2-D PCA with an arrow parent → child per shared
character; each row's weight norm and its bias, parent against child, on the identity line. Classes from the
practice references: frequent (20 most common), rare (under 10 occurrences, or none), other, new (rows only the
child has), blank. Marker shapes repeat the class, because three colour pairs fall below 8 OKLab ΔE under CVD.

![Output layer xix-v1 → xix-blb-v1: rows projected with arrows, row norms and biases parent against child.](outgeom-kraken-german-xix-blb-v1.svg#only-light)
![Output layer xix-v1 → xix-blb-v1: rows projected with arrows, row norms and biases parent against child.](outgeom-kraken-german-xix-blb-v1-dark.svg#only-dark)

![Output layer bern-hertwig-v2 → bern-combined-v1, same panels.](outgeom-kraken-bern-combined-v1.svg#only-light)
![Output layer bern-hertwig-v2 → bern-combined-v1, same panels.](outgeom-kraken-bern-combined-v1-dark.svg#only-dark)

| class | rows (xix / Bern) | median norm, child | share of rows grown | median bias change | median distance moved |
|---|---|---|---|---|---|
| blank | 1 / 1 | 4.6 / 2.8 | 100 % / 0 % | +0.11 / +0.12 | 0.93 / 0.75 |
| frequent | 20 / 20 | 8.8 / 5.3 | 80 % / 10 % | +0.07 / -0.01 | 1.46 / 1.13 |
| other | 48 / 23 | 12.1 / 7.8 | 50 % / 48 % | +0.05 / -0.04 | 2.10 / 1.41 |
| rare | 156 / 223 | 18.1 / 15.8 | 73 % / 87 % | -0.67 / -0.72 | 3.00 / 2.12 |
| new | 72 / 3 | 11.5 / 6.1 | — / 0 % | — / — | — / — |

What the pictures show, in both lineages:

- **Size follows rarity.** Frequent letters have the smallest rows (median norm 8.8 / 5.3), rare characters the
  largest (18.1 / 15.8); the blank is the smallest of all (4.6 / 2.8) and carries the largest bias.
- **The fine-tune works on the rare rows.** They grow (73 % / 87 % of them), lose bias (median −0.67 / −0.72)
  and move furthest (3.0 / 2.1, against 1.5 / 1.1 for frequent letters). In the Bern pair a whole block of rare
  rows loses about one unit of bias together, which is the bias panel's lower band. This is the phase-0
  reading "rare rows inflated" made visible, and it matches R6: rare rows grow and drop their bias in every
  parent → run pair of the seed experiment.
- **The blank's bias rises by about +0.11 in both**, the blank-bias reading of phase 0; the blank's norm grows
  in the Kurrent pair and shrinks in the Bern pair.
- **New rows** (72 in xix-blb-v1, 3 in combined-v1) form their own cluster in the projection, apart from the
  parent's rows: the grown alphabet starts from somewhere else.

## Confusion-to-frame index (08.11.2026)

The page now has a **confusion** selector above the line list, built from every edit of every practice line
(`wsp.vizpage.confusion_index`, edit alignment `wsp.metrics.align`, the same that gives each character its
status): reference → hypothesis, ∅ for nothing, so deletions (`e → ∅`), replacements (`e → c`) and insertions
(`∅ → ␣`). Choosing one filters the list to the lines where it happens, sorted by how often, outlines every
occurrence (insertions as ▾ between characters), and for deletions says at which layer the probes last had the
character. The URL fragment `#conf=<n>` preselects one, so a view can be linked. The same index is committed as
`confusions-<model>__<set>.json` (every occurrence with its line, reference index and frame).

xix-blb-v1 on `blb-train-300`, the five most frequent: ␣ → ∅ 77 (41 lines; deepest layer still having it: output
layer 31, none 23), e → ∅ 32 (none 20), n → ∅ 32 (none 24), ∅ → ␣ 18, i → ∅ 18 (none 13). Practice lines only: the
held-out lines are read outside sessions and are not in the page.

## The Bern family (08.11.2026)

The same tools on `kraken-bern-combined-v1` with its parent `kraken-bern-hertwig-v2`, practice `bern-train-300`
(103 Hertwig and 197 Schlussredaktion lines): `kraken-bern-combined-v1.html`, `probe-…`, `alpha-overlay-…`,
`bias-…`, `outgeom-kraken-bern-combined-v1`, `confusions-…`. One difference: `bern-train-300` comes from only two
Transkribus documents, so the probes are cross-fitted over **5 page folds** (`wsp.layerview --fold-unit page`)
instead of document folds. Test lines then share a hand with training lines, which makes the probe scores
optimistic compared with the Kurrent page; the per-layer *order* is what to compare, not the levels.

| layer | C_0 | C_3 | C_6 | C_9 | L_12 | L_14 | L_16 | O_18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity, letters emitted correctly (7,841) | 0.22 | 0.36 | 0.49 | 0.60 | 0.71 | 0.90 | **0.96** | 0.95 |

Each block's share of the fine-tune (that block alone set back to the parent, practice CER 14.56 %):

| block at α = 0 | ΔCER (points) | letters deleted | spaces deleted | substituted | inserted |
|---|---:|---:|---:|---:|---:|
| C_0 | −0.04 | −10 | −11 | +10 | +7 |
| C_3 | +0.20 | +16 | +7 | +5 | −6 |
| C_6 | +0.23 | −1 | +10 | +19 | −3 |
| C_9 | +0.18 | −4 | +15 | +16 | −7 |
| L_12 | +0.16 | −24 | +14 | +23 | +4 |
| L_14 | +0.14 | −99 | +12 | +81 | +21 |
| **L_16** | **+1.04** | −11 | +10 | **+109** | +5 |
| O_18 | **−0.42** | −118 | −30 | +75 | +27 |

- **A small fine-tune whose specialisation sits in the last LSTM.** Only `L_16` costs more than a quarter point
  when set back (+1.04, mostly substitutions); in the Kurrent pair every recurrent block costs 4–6 points.
- **The output layer goes the other way:** back to the parent's, practice CER falls by 0.42 points (fewer
  deletions, more substitutions). That is the transplant of R3, and as R3 found, the gain comes from the 103
  Hertwig practice lines the parent was trained on; it did not hold on held-out pages.
- **Deletions:** of all deleted characters, 72 % are decodable at no layer (696 of 961), as in the Kurrent pair
  (259 of 361, 72 %); of deleted letters alone 82 % (659 of 807; Kurrent 83 %, 236 of 284). Deleted spaces differ:
  88 of 154 are still decodable up to the output layer and lost only there (Kurrent: 31 of 77).

## Relation to the activation-atlas epic (thodel/weightspace#13)

Epic #13 asks for an atlas at the level of single units: selectivity, occlusion and ablation,
across seeds. This page works one level up, per block and per character.
- Its level L0 (the activation map under the line, with the CTC track) is what the decision views
  and `kraken-german-xix-blb-v1.html` already show.
- The identity and emission probes here are a block-level form of its L1.
- The α overlay is a block-level form of its L3.

The atlas should reuse `wsp.layerview`'s frame map and forced alignment rather than build them
again.

## Epic status (thodel/weightspace#3)

| sub-task | state |
|---|---|
| activation capture | done: `KrakenSubject.activations` (from PR #10) plus the frame map here |
| per-frame decision view | done, as static SVG per line: image, conv channel energy, BiLSTM ‖h‖ forward and reverse, CTC posteriors, reference against hypothesis, probes per layer |
| layer probes | done (this page) |
| ablation and α overlay per line | done: `wsp.alpha_overlay` (above), corpus summary plus per-line strips per block and α |
| output-layer geometry | done (08.11.2026): `wsp.outgeom`, projection plus norm and bias panels, both lineages |
| confusion-to-frame index | done (08.11.2026): selector on the page, `#conf=<n>` links, `confusions-*.json` |
| HTML page | done: `kraken-german-xix-blb-v1.html` and `kraken-bern-combined-v1.html` |
| readings into `docs/READINGS.md` | done: section of 07.11.2026, with four part-block hypotheses for a preregistered session |
