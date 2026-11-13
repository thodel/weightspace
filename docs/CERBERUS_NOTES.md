# CERberus — what this repo takes from it

Source: W. Haverals, *CERberus — guardian against character errors*,
https://github.com/WHaverals/CERberus (read at commit `19cceff`, 07.11.2026; MIT;
Zenodo DOI 10.5281/zenodo.10668052 named in its README). A Flask app around one
function, `cer_module.cer`, that aligns a reference and a hypothesis text with
`Levenshtein.opcodes` and reports CER, per-character statistics, per-Unicode-range
("block") statistics and a confusion table, with optional normalisations.

## The reusable ideas

1. **Report the alignment, not only the distance.** CER is one number; the
   alignment behind it (equal / substitution / insertion / deletion per
   character) can be aggregated any way the question needs. Everything else in
   CERberus is a view of that one alignment. `wsp.metrics.align` already
   produced it for the confusion list; `wsp.blocks` now aggregates it by block.

2. **Aggregate by character class, so errors have a location on the inventory.**
   Per-character tables are too sparse to compare models (most characters occur
   a handful of times); per-block tables are dense enough to carry an interval
   and still say *what kind* of character is lost. This is the step from
   "CER rose by 0.4 points" to "the edit traded 0.6 points of dropped
   diacritics for 0.2 points of inserted spaces", which is exactly what the
   phase-1 and phase-3 sessions needed and did not have.

3. **Normalisation switches as diagnostics.** CERberus lets one ignore case,
   punctuation, whitespace, digits or named characters. Read as a family,
   `CER − CER(ignore X)` is the part of the error that X accounts for. That is a
   coarse, model-free version of the block table; the block table makes it
   additive and gives it intervals. Not ported as switches, because the
   harness's CER must stay the trainer's (NFD, whitespace flattened) for every
   number to stay comparable; the block table answers the same question
   without changing the metric.

4. **Domain ranges next to Unicode blocks.** The MUFI list (medieval special
   characters) cuts across Unicode blocks. Kept here verbatim as an *overlay*
   group reported beside the partition; a project-specific list (abbreviation
   marks of the Bern hands, the BLB's typographic signs) can be added the same
   way.

5. **Line filters as an explicit, counted step.** CERberus can discard lines
   that contain given characters and reports how many it dropped. The harness
   should treat exclusion the same way when the collapsed-lines work proposes
   scoring with and without collapsed lines: never silently, always with the
   count.

## What is not reused, and why

| CERberus | here | why |
|---|---|---|
| counts a character only if it falls in a ticked range | a total partition (ASCII split as in CERberus, then Unicode blocks, then a fallback bucket) | spaces, `ß`, `ſ`, combining marks and every non-ASCII letter drop out otherwise; under NFD the combining diaeresis alone is the most frequent non-ASCII character in our sets (2,725 occurrences) |
| insertions are not attributed to any block (`char1 == '_'` is skipped) | an insertion is charged to the inserted character's block | without it the block numbers do not add up to the CER, and an edit that trades deletions for insertions (every phase-1 lever) looks like a pure gain |
| `'_'` as the gap symbol in the alignment | `''` | a real underscore in the reference would be read as a gap |
| one alignment over the whole text, lines joined by `\n` | one alignment per line | the paired bootstrap resamples lines; a corpus alignment can also pair characters across a line break |
| percentages rounded to two decimals, no uncertainty | raw counts, paired per-line bootstrap per block (2,000 resamples, the protocol's seed) | model comparisons here are differences of a few tenths of a point |
| `Punctuation` = U+0021–002E and U+005B–005F | all ASCII non-alphanumerics, then *General* and *Supplemental Punctuation* as their blocks | `/ : ; ? @` and the en/em dashes, `„`, `⸗` of our transcriptions fall outside its ranges |
| no Unicode normalisation | the harness's NFD | the CER is the trainer's; the block table must describe the same string |

One property to keep in mind when reading block numbers: a minimal-edit
alignment is not unique. When a substitution and a deletion cost the same,
the backtrace's tie order decides which character carries the error. Totals
per block are stable across models scored with the same aligner; individual
attributions near ties are not.

## How it is wired in

- `wsp/blocks.py`: the partition, `line_counts`, `table`, `bootstrap_blocks`.
- `wsp.metrics.score` adds the block table to every `Metrics` (`blocks` field), so baselines,
  sessions, the α-sweep and held-out scoring all carry it from now on.
- `wsp/replay.py`: re-runs a finished session exactly (decisions from its log) so its passes
  get the new statistics; `wsp/blockreport.py`: the tables in docs/BLOCK_STATS.md.
