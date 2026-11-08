# Pre-registration — a third latin_incunabula candidate that protects the abbreviation characters (08.11.2026)

Written and committed before any held-out score of `kraken-latin_incunabula-wsp-v3` (this run). The held-out numbers
known when this was written are R9's for v1 and v2 (docs/RESULTS.md R4, R9) and, from the same stored readings of
R9, their page-clustered intervals (computed today with the clustered bootstrap added in this branch; listed below).
Nothing of v3 has been read on a held-out line.

## Candidate

**v3** = this run: kraken-latin_incunabula with

- `Gn_1` + `Gn_3` from kraken-catmus_print_large (β = 1, as v1 and v2), and
- the output layer `O_15` moved half-way (β = 0.5) **only on the rows of characters both codecs share that are not
  abbreviation characters**: not in the MUFI overlay (CERberus list), not in *Combining Diacritical Marks*, not in
  *Latin Extended-D* (`wsp.exchange.selected_rows(…, "nonabbrev")`). Rows are paired by character
  (`aligned_parent_output`), never by index. The two codecs share 121 characters; 13 of them are protected and keep
  latin_incunabula's rows (U+0303, U+0304, U+0307, U+0313, U+0328, U+0365, ꝑ ꝓ ꝗ ꝙ ꝟ ꝯ, ω); 108 move. The 102 rows
  of characters catmus_print_large lacks keep latin_incunabula's rows, as before.
- **Blank row: moved** with the 108 (as in v1). It is not a character, so the protection rule does not reach it,
  and the variant that keeps it (below) was no better on the abbreviation blocks of the practice lines.
- Not protected although an abbreviation sign: ⁊ (U+204A, *General Punctuation*). The criterion below is
  measured on the three blocks only, so the protection follows the same blocks; ⁊ errors appear in the block table.

Combination `v3-nonabbrev` of `docs/family/combine-v3-latin_incunabula-from-catmus_print_large.json`, rebuilt with
`python -m wsp.exchange --candidate … v3-nonabbrev`: tensor digest `bd0f11ec263f…`, file sha256 `11fffd8a0166…`
(weights not committed).

## How it was chosen (practice and confirmation lines only)

Six combinations, each read on the three practice sets of R4 (`reichenau-val-300`, `gt-fraktur-300`,
`newseye-onb-val-300`) and the three confirmation sets (`…-conf-300`), paired against the unedited model. Two are
references, reproduced bit-exactly (v1 `cc15de58…`, v2 `1fe624af…`); **four variants were tried**, all with
`Gn_1` + `Gn_3`:

| variant | Reichenau Δ, practice / conf | MUFI errors, practice / conf (unedited 32 / 28) | GT-Fraktur Δ, practice / conf | NewsEye Δ, practice / conf |
|---|---|---|---|---|
| v1 (reference: all shared rows, β 0.5) | +0.01 / +0.07 | 46 / 41 | −0.60 / −0.89 | −2.40 / −1.83 |
| v2 (reference: no output move) | +0.06 / +0.07 | 34 / 27 | −0.43 / −0.17 | −0.98 / −0.40 |
| **nonabbrev, blank moved, β 0.5 (v3)** | +0.02 / +0.04 | 34 / 29 | −0.35 / −0.45 | −2.03 / −1.34 |
| nonabbrev, blank kept, β 0.5 | +0.03 / +0.02 | 34 / 29 | −0.40 / −0.55 | −1.91 / −1.09 |
| letters a–z A–Z only (+ blank), β 0.5 | +0.05 / +0.03 | 34 / 29 | −0.13 / −0.13 | −1.47 / −1.07 |
| letters only (+ blank), β 0.25 | +0.03 / +0.05 | 33 / 29 | −0.32 / −0.27 | −1.26 / −0.65 |

(points of CER; every Reichenau interval contains 0; intervals and block tables in the JSON.) On 300-line sets the
four variants cannot be told apart on the abbreviation blocks; all are at v2's level, and v1's MUFI loss
(+14 / +13 errors, interval excluding 0) is gone. The choice among them was by the largest mean Fraktur gain over
the four Fraktur readings: −1.04 (v3) against −0.99, −0.70, −0.63. **Only v3 goes to the held-out sets.**

## Held-out sets, each read once by v3

As R9 (line level, NFD, `wsp.metrics`; unedited readings are R9's stored ones in `docs/baselines/readings/`, not read
again):

- `reichenau-heldout` (4,374 lines, 58 pages, sha256 `bc58cb74…`; sub-collections German 163 lines and
  Latin-German 4,211 lines, reported separately too);
- `newseye-heldout` (2,706 lines, 8 pages, sha256 `98bff6b3…`);
- `gt-fraktur-heldout` (1,311 lines, 35 pages, sha256 `89474aa1…`).

Statistics: paired bootstrap, 2,000 resamples, seed 20261106; for every difference two intervals, **line level**
(`ci95`, R9's) and **page-clustered** (`ci95_cluster`, two-stage: pages with replacement, then lines within the
drawn page); the block statistics of `wsp.blocks` with both intervals. Computation: `python -m wsp.candidates
latin_incunabula-v3` (reads stored readings only).

## Criteria (all in points of CER)

**1. Loses abbreviations** if, on `reichenau-heldout` as a whole or in either sub-collection, for MUFI, Combining
Diacritical Marks or Latin Extended-D, the **line-level** 95 % interval of the change of the block's share of CER
lies entirely above **+0.02 points**.

Why +0.02, from R9's numbers: v1's loss was +0.110 (MUFI, 532 → 731 errors, about one more error every 22 lines),
v2's +0.010 (532 → 551, one every 230 lines). +0.02 points is about 36 more errors in the 181,014 reference
characters of the set, 7 % more MUFI errors, one more every 120 lines: above what v2 showed and a fifth of what v1
lost. It turns R9's "any loss" into "a loss twice v2's point estimate, shown". Why the line-level interval: a harm is
declared on the narrower interval, so the rule errs towards the model's users; the page-clustered interval is
reported beside it. For calibration (known before this was written), R9's intervals in both kinds:

| R9 candidate | block (all lines) | line level | page-clustered |
|---|---|---|---|
| v1 | MUFI | [+0.092, +0.127] | [+0.076, +0.144] |
| v1 | Combining Diacritical Marks | [+0.024, +0.072] | [+0.004, +0.089] |
| v2 | MUFI | [+0.003, +0.019] | [−0.002, +0.024] |
| v2 | Combining Diacritical Marks | [−0.004, +0.024] | [−0.009, +0.031] |

**2. Reichenau worse** if the line-level 95 % interval of the total CER change on `reichenau-heldout` lies entirely
above **+0.05 points** (1 % of the unedited 5.01 %, about 90 errors in the set, one every 50 lines). R9's v2 was
+0.04, line level [+0.01, +0.08], page-clustered [−0.01, +0.10]: a change of that size is not what this criterion
is meant to stop.

**3. Fraktur.** v3 is *worse* on a Fraktur set if the line-level interval of its total CER change lies entirely above
0 (no minimum: a loss on the material the exchange is for defeats its purpose). It is *better* on a Fraktur set if
the **page-clustered** interval lies entirely below 0 (a claim of gain needs the page-level interval). NewsEye has
only 8 pages, so its page-clustered interval is wide (R9's v1: −2.04, pages [−2.42, −1.65]).

## Decision rule

v3 **qualifies** if it does not lose abbreviations (1), Reichenau is not worse (2), it is worse on neither Fraktur set
and better on at least one (3).

- **If v3 qualifies:** the result is written up, and the session **stops and asks** before anything is registered.
  Only with the user's OK is it registered on the share, enabled for the evaluation only, read through the serving
  path on the 58 Reichenau pages as in R4, and disabled again; whether it is then offered stays a further decision of
  the user.
- **If v3 does not qualify:** nothing is registered or enabled. v1 stays registered and disabled (R4, R9). The three
  held-out sets will then have been read by three exchange candidates of this model; a v4 would be designed knowing
  three of their readings, so a further candidate needs fresh held-out lines and its own preregistration.

## What does not change

R9's decisions on v1 and v2 stand and are **not re-judged** under this rule; their numbers appear here only as the
calibration the thresholds were chosen from, which is said openly because it was known.
