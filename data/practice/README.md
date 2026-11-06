# Practice sets (train side only)

Built by `python -m wsp.data build` with seed 20261106 (docs/BRIEF_PHASE0.md §4). Each list is `<set>.jsonl` with `{key, text, image, doc, image_sha256}`; the crops live under `~/atr-cache/wsp/practice/<set>/` (git-ignored). The sha256 below is of the list file, which includes every crop's sha256, so it identifies the pixels as well as the selection. `<set>.meta.json` records source, candidate count and the document ids; `overlap.json` is the audit of `python -m wsp.audit overlap`.

| set | lines | candidates | documents | list sha256 |
|---|---:|---:|---:|---|
| bern-hertwig-train-300-export1025 | 300 | 1984 | 1 | `6123747416ae5e4a45114ad1dcb237f67d2526c430ad956c34212ffe61f15b12` |
| bern-hertwig-train-300 | 300 | 1984 | 1 | `bd4e5a8a4957ac59d2c9126adae1c2006d860543f2b4a9fb94cbc5953a7fe25e` |
| blb-train-300 | 300 | 38724 | 114 | `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b` |
| medieval-train-300 | 300 | 236778 | — | `a2160d4ef170a31e7ab0f3b96d631272347e281a1cbc7f65423f5d0dd8f5dd3d` |
| medieval-val-300-archive | 300 | 20535 | — | `7e3f6eebc08e717e444e3f644b38075152a4abe71550723fad12bba34b9b1290` |
| medieval-val-300 | 300 | 20535 | — | `a2f0f40006e7a377b92cef13ad0274ed5fcf91691c545a86f4309767775e061f` |

## bern-hertwig-train-300-export1025

same lines (same keys, same seed) from the **25.10.2026** export (`59ec6c64`), kept only to document the image mismatch; not a practice set for sessions.

Selection: `random.Random(20261106).sample` over 1984 candidate lines in source order (pages sorted by name, lines in document order; arrow rows by index). Source: `parquet:/home/tobias/atr-cache/wsp/hf/bern-disputation-1528/data/train/A_V_1443_Hertwig_train/20261025_170131-0000.parquet`.

Documents: `13922499`

## bern-hertwig-train-300

HF `dh-unibe/image-text_bern-disputation-1528`, project `A_V_1443_Hertwig_train`, revision `28a16d55` (export of 22.10.2026 — the one hertwig-v1/v2 were trained on; the 25.10 export pairs 44 of 50 pages with the wrong image), lines cut from the Transkribus polygons with `extract_polygons`. Eval pages 8/17/26/39/48/55 are not in this project.

Selection: `random.Random(20261106).sample` over 1984 candidate lines in source order (pages sorted by name, lines in document order; arrow rows by index). Source: `parquet:/home/tobias/atr-cache/wsp/hf/bern-disputation-1528-rev20261022/data/train/A_V_1443_Hertwig_train/20261022_114935-0000.parquet`.

Documents: `13922499`

## blb-train-300

`/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/{lassberg,korrespondenten,unbestimmt,manuskripte}` (the four *train* projects; `_eval` projects and `drucke` excluded), PAGE XML rectangles with bottom-edge baselines, cut with `extract_polygons`. Document = BLB shelfmark.

Selection: `random.Random(20261106).sample` over 38724 candidate lines in source order (pages sorted by name, lines in document order; arrow rows by index). Source: `pagexml:/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/lassberg,/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/korrespondenten,/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/unbestimmt,/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/manuskripte`.

Documents: `K2911,100`, `K2911,101`, `K2911,113`, `K2911,120`, `K2911,122`, `K2911,131`, `K2911,15`, `K2911,16`, `K2911,18`, `K2911,23`, `K2911,24`, `K2911,29`, `K2911,3`, `K2911,40`, `K2911,42`, `K2911,45`, `K2911,49`, `K2911,50`, `K2911,57`, `K2911,68`, `K2911,69`, `K2911,70`, `K2911,71`, `K2911,76`, `K2911,8`, `K2911,80`, `K2911,89`, `K2911,90`, `K2911,92`, `K2911,95`, `K2911,97`, `K2911,98`, `K2912,10`, `K2912,14`, `K2912,17`, `K2912,23`, `K2912,24`, `K2912,25`, `K2912,26`, `K2912,27`, `K2912,3`, `K2912,30`, `K2912,38`, `K2912,40`, `K2912,41`, `K2912,43`, `K2912,44`, `K2912,45`, `K2912,46`, `K2912,47`, `K2912,52`, `K2912,57`, `K2912,6`, `K2912,7`, `K2912,72`, `K2912,74`, `K2912,77`, `K2912,8`, `K2912,9`, `K2913,1`, `K2913,11`, `K2913,12`, `K2913,13`, `K2913,14`, `K2913,18`, `K2913,21`, `K2913,22`, `K2913,23`, `K2913,4`, `K2913,5`, `K2913,7`, `K2913,9`, `K2914,22`, `K2915,13`, `K2915,15`, `K2915,5`, `K2915,6`, `K2915,9`, `K3123,10`, `K3123,11`, `K3123,12`, `K3123,15`, `K3123,17`, `K3123,18`, `K3123,19`, `K3123,24`, `K3123,25`, `K3123,26`, `K3123,28`, `K3123,3`, `K3123,36`, `K3123,37`, `K3123,38`, `K3123,4`, `K3123,40`, `K3123,44`, `K3123,45`, `K3123,46`, `K3123,47`, `K3123,50`, `K3123,52`, `K3123,55`, `K3123,56`, `K3123,56,2`, `K3123,56,5`, `K3123,7`, `K3123,9`, `K3525,6`, `K3525,7`, `K3526`, `K3527`, `K3528`, `K3530,1-3`, `K3533`

## medieval-train-300

`~/atr-cache/arrows/sweep_train.arrow` (train side of `german-medieval-v1`, 236,908 lines, compiled 30.09.2026). Rows carry no document id; held-out documents were removed at compile time by the trainer's held-out boundary.

Selection: `random.Random(20261106).sample` over 236778 candidate lines in source order (pages sorted by name, lines in document order; arrow rows by index). Source: `arrow:/home/tobias/atr-cache/arrows/sweep_train.arrow`.

## medieval-val-300-archive

the same 300 row indices drawn from the *archived* idhefix copy of the validation arrow (`archive/idhefix-2026-09/arrows/german_val.arrow`, a different compile of the same pages); a diagnostic set only, built to check that the asteraix and idhefix arrows agree.

Selection: `random.Random(20261106).sample` over 20535 candidate lines in source order (pages sorted by name, lines in document order; arrow rows by index). Source: `arrow:/mnt/wbkolleg_dh_1/Textrecognition_Training/archive/idhefix-2026-09/arrows/german_val.arrow`.

## medieval-val-300

`~/atr-cache/arrows/german_val.arrow` (the 150 *validation* documents of `german-medieval-v1`, compiled 29.09.2026 on asteraix; sha256 in the meta file). Rows carry no document id. Not the test side (`german_test.arrow` untouched).

Selection: `random.Random(20261106).sample` over 20535 candidate lines in source order (pages sorted by name, lines in document order; arrow rows by index). Source: `arrow:/home/tobias/atr-cache/arrows/german_val.arrow`.
