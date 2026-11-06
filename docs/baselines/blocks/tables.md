## Sessions, pass by pass, and their held-out scores

#### Session `20261106-german-xix-blb-v1-b10-1` — per pass, against the state it was compared with

Replay exact: **True** (pass digests equal True, decisions equal True, final digest equal True, every pass reads the same text on every line). Cells: change of the block's share of CER in points against the last kept state; **bold** where the paired interval excludes 0.

| pass | hypothesis | kept | Δ CER (pt) | Space | a-z | A-Z | Combining marks | ASCII punct. | Latin-1 | Gen. punct. | Digits 0-9 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | H1 blank bias -0.113 | yes | -0.02 | +0.00 | -0.02 | +0.00 | +0.00 | -0.01 | +0.00 | +0.00 | +0.01 |
| 3 | H2 blank bias to -0.5 total | no | +0.06 | -0.02 | +0.08 | +0.00 | -0.01 | +0.00 | +0.00 | +0.01 | +0.00 |
| 4 | H3 space bias +1.0 | no | +0.03 | -0.02 | **+0.04** | -0.01 | +0.00 | +0.02 | +0.00 | +0.00 | +0.00 |
| 5 | H4 space bias +0.5 | no | +0.05 | +0.01 | +0.02 | +0.00 | +0.00 | +0.02 | +0.00 | +0.00 | +0.00 |
| 6 | H5 rare rows from parent | no | +0.02 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.02 | +0.00 |
| 7 | H6 frequent rows x1.1 | no | +0.06 | +0.00 | +0.03 | +0.02 | +0.00 | +0.01 | +0.00 | +0.00 | +0.00 |
| 8 | H7 blank bias -1.0 total | yes | -0.01 | -0.09 | +0.09 | +0.01 | -0.01 | -0.01 | +0.00 | +0.01 | +0.00 |

#### Session `20261106-german-xix-blb-v1-b10-1`: held-out `blb-eval`, frozen artefact against the unedited subject

From the stored held-out readings of the original run; the replay's final digest equals it.

Corpus CER 12.34 % → 12.33 % (-0.01 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| Space | 56700 | 4649 → 4521 | **-0.04** | [-0.05, -0.02] | -0.23 |
| Gen. punct. | 404 | 262 → 256 | -0.00 | [-0.00, +0.00] | -1.49 |
| ASCII punct. | 11636 | 2290 → 2285 | -0.00 | [-0.01, +0.01] | -0.04 |
| Latin-1 | 287 | 101 → 97 | -0.00 | [-0.00, +0.00] | -1.39 |
| MUFI* | 35 | 26 → 22 | **-0.00** | [-0.00, -0.00] | -11.43 |
| Greek and Coptic | 35 | 34 → 34 | +0.00 | [+0.00, +0.00] | +0.00 |
| Superscripts and Subscripts | 10 | 10 → 10 | +0.00 | [+0.00, +0.00] | +0.00 |
| Suppl. punct. | 102 | 102 → 102 | +0.00 | [+0.00, +0.00] | +0.00 |
| Lat.Ext-A | 37 | 37 → 38 | +0.00 | [+0.00, +0.00] | +2.70 |
| Combining marks | 2652 | 808 → 814 | +0.00 | [-0.01, +0.01] | +0.23 |
| Digits 0-9 | 6420 | 1394 → 1408 | +0.00 | [-0.00, +0.01] | +0.22 |
| A-Z | 16705 | 4118 → 4157 | **+0.01** | [+0.01, +0.02] | +0.23 |
| a-z | 242674 | 27831 → 27879 | +0.01 | [-0.01, +0.04] | +0.02 |

#### Session `20261106-german-xix-blb-v1-b10-1`: held-out `blb-eval`, collapsed lines excluded

438 of 5628 lines dropped because one of the two readings collapses on them.

Corpus CER 8.11 % → 8.12 % (+0.01 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| Space | 50920 | 1923 → 1810 | **-0.04** | [-0.05, -0.02] | -0.22 |
| Gen. punct. | 305 | 167 → 161 | -0.00 | [-0.00, +0.00] | -1.97 |
| MUFI* | 32 | 23 → 19 | **-0.00** | [-0.00, -0.00] | -12.50 |
| Latin-1 | 265 | 87 → 85 | -0.00 | [-0.00, +0.00] | -0.75 |
| Greek and Coptic | 32 | 31 → 31 | +0.00 | [+0.00, +0.00] | +0.00 |
| Superscripts and Subscripts | 10 | 10 → 10 | +0.00 | [+0.00, +0.00] | +0.00 |
| Suppl. punct. | 89 | 89 → 89 | +0.00 | [+0.00, +0.00] | +0.00 |
| Lat.Ext-A | 34 | 34 → 35 | +0.00 | [+0.00, +0.00] | +2.94 |
| ASCII punct. | 9946 | 1503 → 1506 | +0.00 | [-0.01, +0.01] | +0.03 |
| Combining marks | 2475 | 682 → 687 | +0.00 | [-0.01, +0.01] | +0.20 |
| Digits 0-9 | 5732 | 998 → 1012 | +0.00 | [-0.00, +0.01] | +0.24 |
| A-Z | 14711 | 2940 → 2980 | **+0.01** | [+0.01, +0.02] | +0.27 |
| a-z | 222042 | 16373 → 16470 | **+0.03** | [+0.01, +0.06] | +0.04 |

#### Session `20261106-bern-combined-v1-b10-1` — per pass, against the state it was compared with

Replay exact: **True** (pass digests equal True, decisions equal True, final digest equal True, every pass reads the same text on every line). Cells: change of the block's share of CER in points against the last kept state; **bold** where the paired interval excludes 0.

| pass | hypothesis | kept | Δ CER (pt) | Space | a-z | A-Z | Combining marks | ASCII punct. | Latin-1 | Gen. punct. | Digits 0-9 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | H1 blank bias to hertwig-v2 | no | -0.06 | -0.04 | -0.04 | +0.00 | +0.00 | +0.02 | +0.00 | +0.00 | +0.00 |
| 3 | H2 O_18 from schlussredaktion-v1 | no | +0.02 | +0.00 | +0.03 | -0.04 | +0.01 | +0.01 | +0.00 | +0.00 | +0.01 |
| 4 | H3 L_16 from schlussredaktion-v1 | no | +0.49 | **+0.18** | +0.07 | +0.01 | **+0.09** | **+0.12** | +0.00 | +0.00 | +0.01 |
| 5 | H4 L_14 from schlussredaktion-v1 | no | +0.57 | **+0.25** | +0.10 | +0.02 | +0.04 | +0.11 | **+0.06** | +0.00 | -0.01 |
| 6 | H5 L_12 from schlussredaktion-v1 | no | -0.07 | +0.16 | -0.17 | -0.01 | +0.02 | -0.06 | +0.01 | +0.00 | -0.03 |
| 7 | H6 C_ from schlussredaktion-v1 | no | +0.52 | **+0.20** | +0.02 | +0.04 | +0.05 | **+0.20** | +0.01 | +0.00 | +0.00 |
| 8 | H7 O_18 from hertwig-v2 | yes | -0.42 | **-0.20** | **-0.55** | **+0.31** | -0.06 | +0.06 | +0.03 | +0.00 | +0.00 |
| 9 | H8 L_16 alpha 0.5 to V2 | no | +0.13 | +0.09 | -0.28 | **+0.13** | **+0.11** | -0.01 | **+0.09** | +0.00 | -0.01 |

#### Session `20261106-bern-combined-v1-b10-1`: held-out `bern-eval`, frozen artefact against the unedited subject

From the stored held-out readings of the original run; the replay's final digest equals it.

Corpus CER 19.33 % → 19.48 % (+0.15 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 12630 | 1971 → 1913 | **-0.35** | [-0.55, -0.15] | -0.46 |
| Digits 0-9 | 25 | 21 → 20 | -0.01 | [-0.03, +0.01] | -4.00 |
| ASCII punct. | 626 | 364 → 364 | +0.00 | [-0.07, +0.07] | +0.00 |
| Latin-1 | 51 | 30 → 32 | +0.01 | [-0.01, +0.04] | +3.92 |
| Space | 2523 | 443 → 447 | +0.02 | [-0.06, +0.11] | +0.16 |
| Combining marks | 245 | 174 → 199 | **+0.15** | [+0.08, +0.22] | +10.20 |
| A-Z | 442 | 192 → 245 | **+0.32** | [+0.22, +0.43] | +11.99 |

#### Session `20261106-bern-combined-v1-b10-1`: held-out `bern-eval`, collapsed lines excluded

38 of 545 lines dropped because one of the two readings collapses on them.

Corpus CER 16.60 % → 16.80 % (+0.20 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 12086 | 1545 → 1493 | **-0.33** | [-0.53, -0.13] | -0.43 |
| ASCII punct. | 589 | 329 → 327 | -0.01 | [-0.09, +0.06] | -0.34 |
| Digits 0-9 | 22 | 18 → 17 | -0.01 | [-0.03, +0.01] | -4.55 |
| Latin-1 | 49 | 28 → 30 | +0.01 | [-0.01, +0.04] | +4.08 |
| Space | 2432 | 368 → 374 | +0.04 | [-0.04, +0.12] | +0.25 |
| Combining marks | 233 | 163 → 188 | **+0.16** | [+0.08, +0.24] | +10.73 |
| A-Z | 421 | 175 → 228 | **+0.33** | [+0.23, +0.45] | +12.59 |

## C3 α-sweep: network block × Unicode block

#### C3 α-sweep `kraken-german-xix-v1` → `kraken-german-xix-blb-v1`: network block × Unicode block

One network block at a time set to parent + α(child − parent), the rest left as the child; practice `blb-train-300`. Cells: change of the Unicode block's share of CER in points against the unedited child (C0); **bold** where the paired 95 % interval excludes 0. At α = 0.5 a positive cell says that network block carries part of the fine-tune's gain *on that kind of character*.

| network block | α | Δ CER (pt) | Space | a-z | A-Z | Combining marks | ASCII punct. | Latin-1 | Gen. punct. | Digits 0-9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_0 | 0.5 | -0.02 | +0.00 | -0.03 | -0.01 | -0.01 | +0.02 | +0.00 | +0.02 | +0.00 |
| C_0 | 1.5 | +0.21 | +0.06 | +0.12 | -0.01 | -0.02 | **+0.05** | +0.00 | +0.00 | +0.01 |
| C_3 | 0.5 | +0.16 | +0.04 | +0.10 | -0.01 | -0.01 | +0.02 | +0.01 | +0.01 | -0.01 |
| C_3 | 1.5 | +0.27 | +0.05 | +0.08 | **+0.07** | +0.02 | **+0.04** | +0.00 | +0.01 | +0.01 |
| C_6 | 0.5 | +0.33 | +0.02 | **+0.25** | +0.04 | +0.00 | +0.02 | +0.00 | +0.00 | +0.00 |
| C_6 | 1.5 | +0.24 | +0.04 | **+0.16** | +0.02 | +0.01 | +0.01 | +0.01 | +0.01 | -0.01 |
| C_9 | 0.5 | +0.16 | +0.04 | +0.12 | -0.01 | -0.01 | +0.00 | -0.01 | +0.00 | +0.02 |
| C_9 | 1.5 | +0.17 | -0.02 | +0.09 | **+0.07** | +0.01 | +0.00 | +0.01 | +0.01 | +0.00 |
| L_12 | 0.5 | +1.90 | **+0.17** | **+1.40** | **+0.12** | +0.02 | **+0.12** | +0.00 | +0.02 | **+0.06** |
| L_12 | 1.5 | +0.27 | -0.01 | +0.18 | **+0.09** | -0.01 | +0.01 | +0.00 | +0.01 | +0.00 |
| L_14 | 0.5 | +1.18 | +0.05 | **+0.79** | **+0.19** | +0.02 | +0.05 | +0.00 | +0.02 | **+0.05** |
| L_14 | 1.5 | +0.37 | +0.08 | **+0.26** | +0.02 | -0.01 | +0.04 | +0.00 | +0.00 | -0.01 |
| L_16 | 0.5 | +2.31 | **+1.37** | **+0.69** | **+0.18** | +0.02 | +0.03 | +0.02 | +0.00 | +0.00 |
| L_16 | 1.5 | +0.05 | +0.02 | -0.05 | +0.05 | +0.00 | +0.02 | +0.00 | +0.01 | +0.02 |
| O_18 | 0.5 | +0.23 | +0.05 | +0.04 | +0.05 | +0.02 | +0.02 | +0.01 | +0.02 | +0.02 |
| O_18 | 1.5 | +0.12 | +0.02 | +0.07 | +0.00 | +0.00 | +0.03 | +0.00 | +0.00 | +0.01 |

## Lineages: paired block deltas between models

#### fine-tune xix-v1 → xix-blb-v1, practice (`blb-train-300`)

Corpus CER 30.39 % → 6.46 % (-23.93 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 9258 | 2525 → 516 | **-15.61** | [-17.63, -13.75] | -21.70 |
| Space | 2201 | 687 → 129 | **-4.34** | [-4.91, -3.82] | -25.35 |
| A-Z | 545 | 280 → 54 | **-1.76** | [-2.07, -1.46] | -41.47 |
| ASCII punct. | 450 | 198 → 64 | **-1.04** | [-1.27, -0.82] | -29.78 |
| Digits 0-9 | 256 | 116 → 36 | **-0.62** | [-0.89, -0.40] | -31.25 |
| Combining marks | 103 | 62 → 15 | **-0.37** | [-0.49, -0.26] | -45.63 |
| Gen. punct. | 17 | 17 → 2 | **-0.12** | [-0.23, -0.04] | -88.24 |
| Latin-1 | 26 | 14 → 5 | **-0.07** | [-0.12, -0.02] | -34.62 |

#### fine-tune xix-v1 → xix-blb-v1, held-out (`blb-eval`)

Corpus CER 31.96 % → 12.34 % (-19.63 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 242674 | 71566 → 27831 | **-12.95** | [-13.35, -12.52] | -18.02 |
| Space | 56700 | 15076 → 4649 | **-3.09** | [-3.22, -2.95] | -18.39 |
| A-Z | 16705 | 10980 → 4118 | **-2.03** | [-2.11, -1.96] | -41.08 |
| ASCII punct. | 11636 | 4572 → 2290 | **-0.68** | [-0.71, -0.64] | -19.61 |
| Digits 0-9 | 6420 | 3144 → 1394 | **-0.52** | [-0.56, -0.48] | -27.26 |
| Combining marks | 2652 | 1820 → 808 | **-0.30** | [-0.32, -0.28] | -38.16 |
| Lat.Ext-A | 37 | 114 → 37 | **-0.02** | [-0.03, -0.02] | -208.11 |
| Gen. punct. | 404 | 336 → 262 | **-0.02** | [-0.03, -0.01] | -18.32 |
| Latin-1 | 287 | 152 → 101 | **-0.02** | [-0.02, -0.01] | -17.77 |
| MUFI* | 35 | 35 → 26 | **-0.00** | [-0.00, -0.00] | -25.71 |
| Greek and Coptic | 35 | 35 → 34 | -0.00 | [-0.00, +0.00] | -2.86 |
| Superscripts and Subscripts | 10 | 10 → 10 | +0.00 | [+0.00, +0.00] | +0.00 |
| Suppl. punct. | 102 | 102 → 102 | +0.00 | [+0.00, +0.00] | +0.00 |

#### fine-tune xix-v1 → xix-blb-v1, held-out (`blb-eval`), collapsed lines excluded

610 of 5628 lines dropped because one of the two readings collapses on them.

Corpus CER 27.62 % → 7.23 % (-20.40 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 216637 | 54220 → 13985 | **-13.47** | [-13.88, -13.07] | -18.57 |
| Space | 49506 | 11509 → 1689 | **-3.29** | [-3.36, -3.21] | -19.84 |
| A-Z | 14355 | 8913 → 2671 | **-2.09** | [-2.17, -2.01] | -43.48 |
| ASCII punct. | 9646 | 3292 → 1366 | **-0.64** | [-0.68, -0.61] | -19.97 |
| Digits 0-9 | 5496 | 2424 → 862 | **-0.52** | [-0.57, -0.48] | -28.42 |
| Combining marks | 2411 | 1600 → 635 | **-0.32** | [-0.35, -0.30] | -40.02 |
| Lat.Ext-A | 33 | 106 → 33 | **-0.02** | [-0.03, -0.02] | -221.21 |
| Gen. punct. | 260 | 196 → 137 | **-0.02** | [-0.03, -0.01] | -22.69 |
| Latin-1 | 235 | 104 → 60 | **-0.01** | [-0.02, -0.01] | -18.72 |
| MUFI* | 33 | 33 → 24 | **-0.00** | [-0.01, -0.00] | -27.27 |
| Greek and Coptic | 35 | 35 → 34 | -0.00 | [-0.00, +0.00] | -2.86 |
| Superscripts and Subscripts | 10 | 10 → 10 | +0.00 | [+0.00, +0.00] | +0.00 |
| Suppl. punct. | 89 | 89 → 89 | +0.00 | [+0.00, +0.00] | +0.00 |

#### hertwig-v2 → combined-v1, practice (`bern-train-300`)

Corpus CER 24.08 % → 14.56 % (-9.52 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 8127 | 1805 → 968 | **-7.73** | [-9.27, -6.18] | -10.30 |
| A-Z | 312 | 181 → 92 | **-0.82** | [-1.07, -0.57] | -28.53 |
| ASCII punct. | 443 | 226 → 188 | **-0.35** | [-0.61, -0.08] | -8.58 |
| Space | 1741 | 268 → 233 | -0.32 | [-0.63, +0.00] | -2.01 |
| Combining marks | 153 | 94 → 73 | **-0.19** | [-0.37, -0.02] | -13.73 |
| Latin-1 | 34 | 22 → 6 | **-0.15** | [-0.23, -0.07] | -47.06 |
| Digits 0-9 | 15 | 9 → 13 | +0.04 | [-0.01, +0.10] | +26.67 |

#### hertwig-v2 → combined-v1, held-out (`bern-eval`)

Corpus CER 27.49 % → 19.33 % (-8.16 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 12630 | 3099 → 1971 | **-6.82** | [-7.82, -5.89] | -8.93 |
| A-Z | 442 | 294 → 192 | **-0.62** | [-0.77, -0.46] | -23.08 |
| Combining marks | 245 | 219 → 174 | **-0.27** | [-0.39, -0.15] | -18.37 |
| Space | 2523 | 480 → 443 | **-0.22** | [-0.43, -0.01] | -1.47 |
| ASCII punct. | 626 | 396 → 364 | **-0.19** | [-0.37, -0.01] | -5.11 |
| Latin-1 | 51 | 37 → 30 | -0.04 | [-0.09, +0.01] | -13.73 |
| Digits 0-9 | 25 | 20 → 21 | +0.01 | [-0.03, +0.04] | +4.00 |

#### hertwig-v2 → combined-v1, held-out (`bern-eval`), collapsed lines excluded

38 of 545 lines dropped because one of the two readings collapses on them.

Corpus CER 25.33 % → 16.60 % (-8.73 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 12072 | 2696 → 1539 | **-7.31** | [-8.30, -6.32] | -9.58 |
| A-Z | 421 | 277 → 176 | **-0.64** | [-0.79, -0.48] | -23.99 |
| Combining marks | 233 | 207 → 163 | **-0.28** | [-0.40, -0.16] | -18.88 |
| Space | 2431 | 411 → 369 | **-0.27** | [-0.48, -0.05] | -1.73 |
| ASCII punct. | 589 | 361 → 329 | **-0.20** | [-0.41, -0.01] | -5.43 |
| Latin-1 | 49 | 35 → 28 | -0.04 | [-0.10, +0.01] | -14.29 |
| Digits 0-9 | 22 | 17 → 19 | +0.01 | [-0.02, +0.05] | +9.09 |

#### schlussredaktion-v1 (clean sibling) → combined-v1, held-out (`bern-eval`)

Corpus CER 20.36 % → 19.33 % (-1.03 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 12630 | 2082 → 1971 | **-0.67** | [-1.06, -0.29] | -0.88 |
| ASCII punct. | 626 | 408 → 364 | **-0.27** | [-0.45, -0.09] | -7.03 |
| Space | 2523 | 472 → 443 | -0.18 | [-0.36, +0.02] | -1.15 |
| Combining marks | 245 | 185 → 174 | -0.07 | [-0.16, +0.02] | -4.49 |
| Latin-1 | 51 | 31 → 30 | -0.01 | [-0.03, +0.02] | -1.96 |
| Digits 0-9 | 25 | 18 → 21 | +0.02 | [-0.01, +0.05] | +12.00 |
| A-Z | 442 | 170 → 192 | **+0.13** | [+0.03, +0.24] | +4.98 |

#### schlussredaktion-v1 (clean sibling) → combined-v1, held-out (`bern-eval`), collapsed lines excluded

37 of 545 lines dropped because one of the two readings collapses on them.

Corpus CER 17.68 % → 16.46 % (-1.22 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 12036 | 1649 → 1519 | **-0.82** | [-1.20, -0.46] | -1.08 |
| ASCII punct. | 585 | 369 → 326 | **-0.27** | [-0.45, -0.09] | -7.35 |
| Space | 2421 | 394 → 364 | -0.19 | [-0.37, +0.00] | -1.24 |
| Combining marks | 231 | 172 → 161 | -0.07 | [-0.16, +0.02] | -4.76 |
| Latin-1 | 49 | 29 → 28 | -0.01 | [-0.03, +0.02] | -2.04 |
| Digits 0-9 | 25 | 18 → 21 | +0.02 | [-0.01, +0.05] | +12.00 |
| A-Z | 420 | 154 → 174 | **+0.13** | [+0.03, +0.23] | +4.76 |

#### cursive_inzigkofen → hertwig-v2, practice (22.10 images) (`bern-hertwig-train-300`)

Corpus CER 70.59 % → 8.28 % (-62.31 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| a-z | 6636 | 4826 → 453 | **-49.28** | [-51.05, -47.57] | -65.90 |
| Space | 1375 | 470 → 101 | **-4.16** | [-4.62, -3.70] | -26.84 |
| ASCII punct. | 393 | 375 → 94 | **-3.17** | [-3.50, -2.82] | -71.50 |
| A-Z | 270 | 275 → 42 | **-2.63** | [-2.98, -2.27] | -86.30 |
| Combining marks | 152 | 179 → 28 | **-1.70** | [-2.00, -1.43] | -99.34 |
| Latin-1 | 17 | 72 → 4 | **-0.77** | [-0.97, -0.57] | -400.00 |
| Lat.Ext-A | 0 | 35 → 0 | **-0.39** | [-0.56, -0.25] |  |
| Digits 0-9 | 26 | 26 → 10 | **-0.18** | [-0.30, -0.08] | -61.54 |
| MUFI* | 0 | 13 → 0 | **-0.15** | [-0.23, -0.07] |  |

#### hertwig-v2 → combined-v1 on the mispaired 25.10 images (`bern-hertwig-train-300-export1025`)

Corpus CER 75.72 % → 85.93 % (+10.21 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| A-Z | 270 | 264 → 245 | **-0.21** | [-0.34, -0.10] | -7.04 |
| Digits 0-9 | 26 | 25 → 24 | -0.01 | [-0.04, +0.00] | -3.85 |
| Latin-1 | 17 | 15 → 15 | +0.00 | [+0.00, +0.00] | +0.00 |
| Combining marks | 152 | 127 → 139 | **+0.14** | [+0.05, +0.24] | +7.89 |
| ASCII punct. | 393 | 337 → 357 | **+0.23** | [+0.10, +0.36] | +5.09 |
| Space | 1375 | 771 → 1149 | **+4.26** | [+3.71, +4.81] | +27.49 |
| a-z | 6636 | 5175 → 5691 | **+5.81** | [+4.82, +6.80] | +7.78 |

#### R1 lossless rewrite Cr255 → Cr1 (`medieval-val-300`)

Corpus CER 99.00 % → 99.00 % (+0.00 points). Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.

| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |
|---|---:|---:|---:|---|---:|
| ASCII punct. | 235 | 227 → 227 | +0.00 | [+0.00, +0.00] | +0.00 |
| Combining marks | 165 | 165 → 165 | +0.00 | [+0.00, +0.00] | +0.00 |
| Digits 0-9 | 89 | 89 → 89 | +0.00 | [+0.00, +0.00] | +0.00 |
| Dingbats | 72 | 72 → 72 | +0.00 | [+0.00, +0.00] | +0.00 |
| Lat.Ext-D | 10 | 10 → 10 | +0.00 | [+0.00, +0.00] | +0.00 |
| a-z | 10512 | 10388 → 10388 | +0.00 | [+0.00, +0.00] | +0.00 |
| A-Z | 285 | 282 → 282 | +0.00 | [+0.00, +0.00] | +0.00 |
| Latin-1 | 11 | 11 → 11 | +0.00 | [+0.00, +0.00] | +0.00 |
| MUFI* | 62 | 62 → 62 | +0.00 | [+0.00, +0.00] | +0.00 |
| Space | 2056 | 2056 → 2056 | +0.00 | [+0.00, +0.00] | +0.00 |
| Modifier letters | 18 | 18 → 18 | +0.00 | [+0.00, +0.00] | +0.00 |

## Appendix: block tables per set and model

#### blb-train-300 — 300 lines, 12,867 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | german-xix-blb-v1 | german-xix-v1 |
|---|---:|---:|---:|
| a-z | 9258 | 5.6 % (4.01) | 27.3 % (19.62) |
| Space | 2201 | 5.9 % (1.00) | 31.2 % (5.34) |
| A-Z | 545 | 9.9 % (0.42) | 51.4 % (2.18) |
| ASCII punct. | 450 | 14.2 % (0.50) | 44.0 % (1.54) |
| Digits 0-9 | 256 | 14.1 % (0.28) | 45.3 % (0.90) |
| Combining marks | 103 | 14.6 % (0.12) | 60.2 % (0.48) |
| Gen. punct. | 17 | 11.8 % (0.02) | 100.0 % (0.13) |
| Latin-1 | 26 | 19.2 % (0.04) | 53.8 % (0.11) |
| **corpus CER** | | **6.46 %** | **30.39 %** |

#### blb-eval — 5628 lines, 337,684 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | german-xix-blb-v1 | german-xix-v1 |
|---|---:|---:|---:|
| a-z | 242674 | 11.5 % (8.24) | 29.5 % (21.19) |
| Space | 56700 | 8.2 % (1.38) | 26.6 % (4.46) |
| A-Z | 16705 | 24.7 % (1.22) | 65.7 % (3.25) |
| ASCII punct. | 11636 | 19.7 % (0.68) | 39.3 % (1.35) |
| Digits 0-9 | 6420 | 21.7 % (0.41) | 49.0 % (0.93) |
| Combining marks | 2652 | 30.5 % (0.24) | 68.6 % (0.54) |
| Gen. punct. | 404 | 64.9 % (0.08) | 83.2 % (0.10) |
| Latin-1 | 287 | 35.2 % (0.03) | 53.0 % (0.05) |
| Lat.Ext-A | 37 | 100.0 % (0.01) | 308.1 % (0.03) |
| Suppl. punct. | 102 | 100.0 % (0.03) | 100.0 % (0.03) |
| Greek and Coptic | 35 | 97.1 % (0.01) | 100.0 % (0.01) |
| MUFI* | 35 | 74.3 % (0.01) | 100.0 % (0.01) |
| Superscripts and Subscripts | 10 | 100.0 % (0.00) | 100.0 % (0.00) |
| **corpus CER** | | **12.34 %** | **31.96 %** |

#### bern-train-300 — 300 lines, 10,829 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | bern-combined-v1 | bern-hertwig-v2 | bern-schlussredaktion-v1 | cursive_inzigkofen |
|---|---:|---:|---:|---:|---:|
| a-z | 8127 | 11.9 % (8.94) | 22.2 % (16.67) | 12.2 % (9.12) | 71.6 % (53.70) |
| Space | 1741 | 13.4 % (2.15) | 15.4 % (2.47) | 15.5 % (2.48) | 27.7 % (4.45) |
| ASCII punct. | 443 | 42.4 % (1.74) | 51.0 % (2.09) | 45.1 % (1.85) | 88.3 % (3.61) |
| A-Z | 312 | 29.5 % (0.85) | 58.0 % (1.67) | 28.8 % (0.83) | 97.1 % (2.80) |
| Combining marks | 153 | 47.7 % (0.67) | 61.4 % (0.87) | 46.4 % (0.66) | 114.4 % (1.62) |
| Latin-1 | 34 | 17.6 % (0.06) | 64.7 % (0.20) | 26.5 % (0.08) | 211.8 % (0.66) |
| Lat.Ext-A | 0 |  |  |  | ins only (0.22) |
| Digits 0-9 | 15 | 86.7 % (0.12) | 60.0 % (0.08) | 60.0 % (0.08) | 100.0 % (0.14) |
| MUFI* | 1 | 100.0 % (0.01) | 0.0 % (0.00) | 100.0 % (0.01) | 1100.0 % (0.10) |
| **corpus CER** | | **14.56 %** | **24.08 %** | **15.14 %** | **67.24 %** |

#### bern-eval — 545 lines, 16,545 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | bern-combined-v1 | bern-hertwig-v2 | bern-schlussredaktion-v1 | cursive_inzigkofen |
|---|---:|---:|---:|---:|---:|
| a-z | 12630 | 15.6 % (11.91) | 24.5 % (18.73) | 16.5 % (12.58) | 73.4 % (56.01) |
| Space | 2523 | 17.6 % (2.68) | 19.0 % (2.90) | 18.7 % (2.85) | 28.9 % (4.41) |
| ASCII punct. | 626 | 58.1 % (2.20) | 63.3 % (2.39) | 65.2 % (2.47) | 89.0 % (3.37) |
| A-Z | 442 | 43.4 % (1.16) | 66.5 % (1.78) | 38.5 % (1.03) | 101.1 % (2.70) |
| Combining marks | 245 | 71.0 % (1.05) | 89.4 % (1.32) | 75.5 % (1.12) | 123.3 % (1.83) |
| Latin-1 | 51 | 58.8 % (0.18) | 72.5 % (0.22) | 60.8 % (0.19) | 233.3 % (0.72) |
| Lat.Ext-A | 0 |  |  |  | ins only (0.24) |
| MUFI* | 3 | 100.0 % (0.02) | 100.0 % (0.02) | 100.0 % (0.02) | 900.0 % (0.16) |
| Digits 0-9 | 25 | 84.0 % (0.13) | 80.0 % (0.12) | 72.0 % (0.11) | 100.0 % (0.15) |
| **corpus CER** | | **19.33 %** | **27.49 %** | **20.36 %** | **69.48 %** |

#### bern-hertwig-train-300 — 300 lines, 8,874 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | bern-combined-v1 | bern-hertwig-v1 | bern-hertwig-v2 | bern-schlussredaktion-v1 | cursive_inzigkofen |
|---|---:|---:|---:|---:|---:|---:|
| a-z | 6636 | 14.9 % (11.11) | 13.2 % (9.85) | 6.8 % (5.10) | 16.0 % (11.99) | 72.7 % (54.38) |
| Space | 1375 | 17.5 % (2.70) | 17.0 % (2.64) | 7.3 % (1.14) | 23.2 % (3.59) | 34.2 % (5.30) |
| ASCII punct. | 393 | 46.1 % (2.04) | 48.6 % (2.15) | 23.9 % (1.06) | 66.4 % (2.94) | 95.4 % (4.23) |
| A-Z | 270 | 35.9 % (1.09) | 43.7 % (1.33) | 15.6 % (0.47) | 40.7 % (1.24) | 101.9 % (3.10) |
| Combining marks | 152 | 70.4 % (1.21) | 71.7 % (1.23) | 18.4 % (0.32) | 74.3 % (1.27) | 117.8 % (2.02) |
| Latin-1 | 17 | 41.2 % (0.08) | 82.4 % (0.16) | 23.5 % (0.05) | 52.9 % (0.10) | 423.5 % (0.81) |
| Lat.Ext-A | 0 |  |  |  |  | ins only (0.39) |
| Digits 0-9 | 26 | 80.8 % (0.24) | 80.8 % (0.24) | 38.5 % (0.11) | 61.5 % (0.18) | 100.0 % (0.29) |
| MUFI* | 0 |  |  |  |  | ins only (0.15) |
| **corpus CER** | | **18.53 %** | **17.65 %** | **8.28 %** | **21.38 %** | **70.59 %** |

#### bern-hertwig-train-300-export1025 — 300 lines, 8,874 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | bern-combined-v1 | bern-hertwig-v1 | bern-hertwig-v2 | bern-schlussredaktion-v1 | cursive_inzigkofen |
|---|---:|---:|---:|---:|---:|---:|
| a-z | 6636 | 85.8 % (64.13) | 78.0 % (58.33) | 78.0 % (58.32) | 81.3 % (60.80) | 87.6 % (65.53) |
| Space | 1375 | 83.6 % (12.95) | 60.5 % (9.38) | 56.1 % (8.69) | 59.4 % (9.21) | 60.1 % (9.32) |
| ASCII punct. | 393 | 90.8 % (4.02) | 88.5 % (3.92) | 85.8 % (3.80) | 92.9 % (4.11) | 97.5 % (4.32) |
| A-Z | 270 | 90.7 % (2.76) | 96.7 % (2.94) | 97.8 % (2.97) | 99.3 % (3.02) | 102.6 % (3.12) |
| Combining marks | 152 | 91.4 % (1.57) | 94.7 % (1.62) | 83.6 % (1.43) | 94.7 % (1.62) | 106.6 % (1.83) |
| Latin-1 | 17 | 88.2 % (0.17) | 100.0 % (0.19) | 88.2 % (0.17) | 94.1 % (0.18) | 217.6 % (0.42) |
| Digits 0-9 | 26 | 92.3 % (0.27) | 100.0 % (0.29) | 96.2 % (0.28) | 100.0 % (0.29) | 100.0 % (0.29) |
| Lat.Ext-A | 0 |  |  |  |  | ins only (0.14) |
| **corpus CER** | | **85.93 %** | **76.73 %** | **75.72 %** | **79.29 %** | **85.01 %** |

#### medieval-train-300 — 300 lines, 13,107 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | collapse-probe-131-cr255 | cursive_inzigkofen | medieval-shard00-std | medieval-shard00-v1 | plus-shard00-cr1 |
|---|---:|---:|---:|---:|---:|---:|
| a-z | 10265 | 98.2 % (76.91) | 77.6 % (60.79) | 63.3 % (49.55) | 64.5 % (50.51) | 65.2 % (51.03) |
| Space | 1976 | 100.0 % (15.08) | 33.2 % (5.01) | 44.3 % (6.68) | 35.2 % (5.30) | 41.0 % (6.18) |
| A-Z | 235 | 99.1 % (1.78) | 110.2 % (1.98) | 102.1 % (1.83) | 100.0 % (1.79) | 100.0 % (1.79) |
| ASCII punct. | 244 | 96.3 % (1.79) | 103.7 % (1.93) | 100.8 % (1.88) | 99.2 % (1.85) | 100.0 % (1.86) |
| Combining marks | 186 | 100.0 % (1.42) | 117.7 % (1.67) | 100.0 % (1.42) | 99.5 % (1.41) | 100.0 % (1.42) |
| Dingbats | 94 | 100.0 % (0.72) | 100.0 % (0.72) | 100.0 % (0.72) | 97.9 % (0.70) | 95.7 % (0.69) |
| Latin-1 | 24 | 75.0 % (0.14) | 350.0 % (0.64) | 100.0 % (0.18) | 100.0 % (0.18) | 100.0 % (0.18) |
| MUFI* | 70 | 100.0 % (0.53) | 110.0 % (0.59) | 100.0 % (0.53) | 100.0 % (0.53) | 100.0 % (0.53) |
| Digits 0-9 | 48 | 100.0 % (0.37) | 100.0 % (0.37) | 135.4 % (0.50) | 100.0 % (0.37) | 95.8 % (0.35) |
| Lat.Ext-A | 1 | 100.0 % (0.01) | 4600.0 % (0.35) | 100.0 % (0.01) | 100.0 % (0.01) | 100.0 % (0.01) |
| Modifier letters | 12 | 100.0 % (0.09) | 100.0 % (0.09) | 100.0 % (0.09) | 100.0 % (0.09) | 100.0 % (0.09) |
| Lat.Ext-D | 9 | 100.0 % (0.07) | 122.2 % (0.08) | 100.0 % (0.07) | 100.0 % (0.07) | 100.0 % (0.07) |
| **corpus CER** | | **98.46 %** | **73.75 %** | **63.01 %** | **62.39 %** | **63.78 %** |

#### medieval-val-300 — 300 lines, 13,469 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | collapse-probe-131-cr1 | collapse-probe-131-cr255 | cursive_inzigkofen | german-xix-blb-v1 | medieval-shard00-std | medieval-shard00-v1 | plus-shard00-cr1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a-z | 10512 | 98.8 % (77.13) | 98.8 % (77.13) | 77.3 % (60.29) | 57.0 % (44.48) | 68.2 % (53.20) | 72.7 % (56.72) | 72.9 % (56.93) |
| Space | 2056 | 100.0 % (15.26) | 100.0 % (15.26) | 36.8 % (5.61) | 26.8 % (4.09) | 49.9 % (7.62) | 48.6 % (7.42) | 56.6 % (8.63) |
| A-Z | 285 | 98.9 % (2.09) | 98.9 % (2.09) | 113.7 % (2.41) | 83.9 % (1.77) | 100.4 % (2.12) | 100.0 % (2.12) | 98.9 % (2.09) |
| ASCII punct. | 235 | 96.6 % (1.69) | 96.6 % (1.69) | 103.8 % (1.81) | 87.2 % (1.52) | 100.0 % (1.74) | 98.7 % (1.72) | 99.1 % (1.73) |
| Combining marks | 165 | 100.0 % (1.23) | 100.0 % (1.23) | 124.2 % (1.52) | 105.5 % (1.29) | 100.0 % (1.23) | 100.0 % (1.23) | 100.0 % (1.23) |
| Digits 0-9 | 89 | 100.0 % (0.66) | 100.0 % (0.66) | 100.0 % (0.66) | 66.3 % (0.44) | 88.8 % (0.59) | 98.9 % (0.65) | 88.8 % (0.59) |
| Dingbats | 72 | 100.0 % (0.53) | 100.0 % (0.53) | 100.0 % (0.53) | 100.0 % (0.53) | 100.0 % (0.53) | 100.0 % (0.53) | 97.2 % (0.52) |
| MUFI* | 62 | 100.0 % (0.46) | 100.0 % (0.46) | 104.8 % (0.48) | 101.6 % (0.47) | 100.0 % (0.46) | 100.0 % (0.46) | 100.0 % (0.46) |
| Latin-1 | 11 | 100.0 % (0.08) | 100.0 % (0.08) | 518.2 % (0.42) | 100.0 % (0.08) | 100.0 % (0.08) | 100.0 % (0.08) | 100.0 % (0.08) |
| Lat.Ext-A | 0 |  |  | ins only (0.27) |  |  |  |  |
| Modifier letters | 18 | 100.0 % (0.13) | 100.0 % (0.13) | 100.0 % (0.13) | 100.0 % (0.13) | 100.0 % (0.13) | 100.0 % (0.13) | 100.0 % (0.13) |
| Lat.Ext-D | 10 | 100.0 % (0.07) | 100.0 % (0.07) | 120.0 % (0.09) | 100.0 % (0.07) | 100.0 % (0.07) | 100.0 % (0.07) | 100.0 % (0.07) |
| **corpus CER** | | **99.00 %** | **99.00 %** | **73.88 %** | **54.54 %** | **67.44 %** | **70.80 %** | **72.13 %** |

#### medieval-val-300-archive — 300 lines, 13,600 reference characters

Per block: reference characters, then for each model the block's own CER (errors / its reference characters) and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).

| block | ref | medieval-shard00-std | plus-shard00-cr1 |
|---|---:|---:|---:|
| a-z | 10578 | 67.8 % (52.76) | 71.2 % (55.39) |
| Space | 2073 | 48.1 % (7.33) | 51.7 % (7.88) |
| A-Z | 309 | 99.4 % (2.26) | 98.7 % (2.24) |
| ASCII punct. | 253 | 100.0 % (1.86) | 100.4 % (1.87) |
| Combining marks | 186 | 100.0 % (1.37) | 100.0 % (1.37) |
| Dingbats | 95 | 100.0 % (0.70) | 100.0 % (0.70) |
| MUFI* | 52 | 100.0 % (0.38) | 100.0 % (0.38) |
| Digits 0-9 | 53 | 92.5 % (0.36) | 94.3 % (0.37) |
| Modifier letters | 22 | 100.0 % (0.16) | 100.0 % (0.16) |
| Latin-1 | 12 | 100.0 % (0.09) | 100.0 % (0.09) |
| **corpus CER** | | **67.03 %** | **70.20 %** |

