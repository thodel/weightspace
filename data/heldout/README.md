# Held-out sets (never practice)

Built by `python -m wsp.data build --heldout --n 0`: every line of the source, no sampling. Each list is `<set>.jsonl` with `{key, text, image, doc, image_sha256}`; crops under `~/atr-cache/wsp/heldout/<set>/` (git-ignored). `<set>.meta.json` records source, candidates, skipped lines and documents; `overlap.json` is the audit of `python -m wsp.audit heldout`. Lines here are scored once per session after `freeze()` (PROTOCOL.md §3) and by `wsp.baselines` for C0/C1; nothing here is ever a practice line.

| set | lines | candidates | documents | list sha256 |
|---|---:|---:|---:|---|
| blb-eval | 5628 | 5644 | 33 | `0729b93b1b5d594d40eeea80ae498330118c3efdbd4f4bbf73ed317059c5dab1` |
| bern-eval | 545 | 545 | 2 | `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227` |
| **blb-eval-hand** | 2272 | 5628 | 29 | `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091` |

## blb-eval

The four handwriting `_eval` projects of `/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt` (lassberg_eval 446 lines / 7 objects, korrespondenten_eval 452 / 8, unbestimmt_eval 193 / 4, manuskripte_eval 4553 / 14); `drucke_eval` left out because `drucke` was not trained on. PAGE XML rectangles with bottom-edge baselines, cut with `extract_polygons` as the practice set. 16 lines whose baseline is shorter than 5 px are skipped (kraken refuses the cut); their keys are in the meta file. Document = BLB shelfmark; every one is an `export.json` eval object (object-level split, seed 20261023), none occurs in any practice list (`overlap.json`, 06.11.2026). The same projects are the trainer's held-out side for `kraken-german-xix-blb-v1` (docs/MODELS.md, 8.7 % CER) and the gateway's `blb-eval-*` runs.

## bern-eval

Family 1's held-out set: the 6 pages of `A_V_1443_Hertwig_test` (doc 13922499, pages 8/17/26/39/48/55, 287 lines) from the **22.10 export** (`28a16d55`, the only one whose images match their XML; docs/HERTWIG_REVISIONS.md) and the 8 pages of `A_V_1447_Schlussredaktion_test` (doc 13922579, pages 12/30/49/69/89/105/125/142, 258 lines; the 24.10 and 25.10 exports are byte-identical), from `dh-unibe/image-text_bern-disputation-1528`. Lines cut from the Transkribus polygons with `extract_polygons`; none skipped. The split is at page level inside one volume per project, so the audit (`overlap-bern-eval.json`, `wsp.audit heldout`) checks (document, page): every row comes from a `_test` project, no page occurs in any practice list, and none of the 14 pages is among the 168 pages of the two training projects (`A_V_1443_Hertwig_train` 50, `A_V_1447_Schlussredaktion_train` 118), i.e. the data `combined-v1` was trained on.

## blb-eval-hand — family 2's held-out set from 07.11.2026 on

The handwriting part of `blb-eval`, chosen by data properties only (`scripts/build_blb_eval_hand.py`): a subset with the same keys, crops and image hashes, so every reading of `blb-eval` re-aggregates on it exactly. No rule looks at a model's output. Excluded objects: K 2913,17 (a printed book whose reference is the library's OCR, 2,825 lines), K 3123,57 (typescript, 372), K 2911,129 (printed form, 21), K 2912,66 (music notation, 12); excluded page: K 2915-3 p. 10 (outside of a folded letter, 12). Excluded lines: crop taller than wide (103), box over 2.5 times the page's median line height (2), reference of 1–3 characters (9); crops under 8 px high were all in the printed book. Kept: 2,272 lines of 29 objects (lassberg 438, korrespondenten 394, unbestimmt 171, manuskripte 1,269). Audit `overlap-blb-eval-hand.json`: clean against every practice list. **Every family-2 number from 07.11.2026 on is measured on this set**; `blb-eval` stays as it is, for continuity with the reports before that date.
