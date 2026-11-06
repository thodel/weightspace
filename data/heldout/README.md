# Held-out sets (never practice)

Built by `python -m wsp.data build --heldout --n 0`: every line of the source, no sampling. Each list is `<set>.jsonl` with `{key, text, image, doc, image_sha256}`; crops under `~/atr-cache/wsp/heldout/<set>/` (git-ignored). `<set>.meta.json` records source, candidates, skipped lines and documents; `overlap.json` is the audit of `python -m wsp.audit heldout`. Lines here are scored once per session after `freeze()` (PROTOCOL.md §3) and by `wsp.baselines` for C0/C1; nothing here is ever a practice line.

| set | lines | candidates | documents | list sha256 |
|---|---:|---:|---:|---|
| blb-eval | 5628 | 5644 | 33 | `0729b93b1b5d594d40eeea80ae498330118c3efdbd4f4bbf73ed317059c5dab1` |

## blb-eval

The four handwriting `_eval` projects of `/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt` (lassberg_eval 446 lines / 7 objects, korrespondenten_eval 452 / 8, unbestimmt_eval 193 / 4, manuskripte_eval 4553 / 14); `drucke_eval` left out because `drucke` was not trained on. PAGE XML rectangles with bottom-edge baselines, cut with `extract_polygons` as the practice set. 16 lines whose baseline is shorter than 5 px are skipped (kraken refuses the cut); their keys are in the meta file. Document = BLB shelfmark; every one is an `export.json` eval object (object-level split, seed 20261023), none occurs in any practice list (`overlap.json`, 06.11.2026). The same projects are the trainer's held-out side for `kraken-german-xix-blb-v1` (docs/MODELS.md, 8.7 % CER) and the gateway's `blb-eval-*` runs.
