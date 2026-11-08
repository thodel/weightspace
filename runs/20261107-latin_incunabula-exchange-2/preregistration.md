# Pre-registration — which latin_incunabula candidate to enable (07.11.2026)

Written before any held-out score of `kraken-latin_incunabula-wsp-v2` (this run) and before any Fraktur held-out
score of either candidate. The only held-out numbers known when this was written are v1's on `reichenau-heldout`
(docs/RESULTS.md R4).

**Candidates.** v1 = `runs/20261107-latin_incunabula-exchange-1` (`Gn_1` + `Gn_3` from kraken-catmus_print_large,
`O_15` half-way). v2 = this run (`Gn_1` + `Gn_3` only; combination `gn13` of
`docs/family/combine-latin_incunabula-from-catmus_print_large.json`, tensor digest `1fe624af…`).

**Held-out sets, each scored once per model** (line level, NFD, `wsp.metrics`; paired per-line bootstrap, 2,000
resamples, seed 20261106; block statistics of `wsp.blocks`):

- `reichenau-heldout` (4,374 lines, 58 pages; German and Latin-German sub-collections, reported separately too);
- `newseye-heldout` (2,706 lines, 8 pages of the NewsEye ONB train shard);
- `gt-fraktur-heldout` (1,311 lines, the 35 GT-Fraktur pages no practice or confirmation line came from).

**Abbreviation criterion.** A candidate *loses abbreviations* if, on `reichenau-heldout` as a whole or in either
sub-collection, the paired change of the error share of **MUFI** (overlay), **Combining Diacritical Marks** or
**Latin Extended-D** has a 95 % interval entirely above 0.

**Decision.** A candidate qualifies if it does not lose abbreviations, its total CER on `reichenau-heldout` is not
worse (interval not entirely above 0), and its total CER on neither Fraktur set is worse. Of the qualifying
candidates, the one with the larger mean CER gain over the two Fraktur sets is read through the serving path on
the 58 Reichenau pages (as R4) and enabled in the registry if the serving path is not worse there either. If no
candidate qualifies, none is enabled.
