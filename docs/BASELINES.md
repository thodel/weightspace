# Baselines — C0, practice sets (not results)

Practice numbers on the *train-side* practice sets of docs/BRIEF_PHASE0.md §4, scored once per model outside any session (PROTOCOL §4, C0). They are **not** held-out results and must not be quoted as such; they fix the starting point every edit is measured against. CER/WER corpus-level, NFD, whitespace flattened (`wsp.metrics`). Blank share: mean softmax mass on the CTC blank over all frames. CPU, kraken 7.0.2, 06.11.2026.

## bern-hertwig-train-300-export1025 — 300 lines, list sha256 `6123747416ae5e4a45114ad1dcb237f67d2526c430ad956c34212ffe61f15b12`

| model | CER | WER | collapsed | length ratio | blank share | params | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kraken-cursive_inzigkofen | 0.8501 | 1.1075 | 134 | 0.623 | 0.787 | 4,097,926 | 18.0 |
| kraken-bern-hertwig-v1 | 0.7673 | 0.9576 | 150 | 0.578 | 0.769 | 4,099,931 | 15.1 |
| kraken-bern-hertwig-v2 | 0.7572 | 0.9313 | 136 | 0.650 | 0.766 | 4,099,931 | 14.3 |
| kraken-bern-schlussredaktion-v1 | 0.7929 | 0.9940 | 133 | 0.647 | 0.795 | 4,100,733 | 14.3 |
| kraken-bern-combined-v1 | 0.8593 | 0.9087 | 267 | 0.154 | 0.778 | 4,101,134 | 15.0 |

- **kraken-cursive_inzigkofen** top confusions (ref→hyp): ` `→`∅` 480, `e`→`∅` 360, `n`→`∅` 283, `t`→`∅` 250, `r`→`∅` 219, `s`→`∅` 207, `d`→`∅` 180, `i`→`∅` 179, `a`→`∅` 163, `l`→`∅` 127
- **kraken-bern-hertwig-v1** top confusions (ref→hyp): ` `→`∅` 578, `e`→`∅` 375, `n`→`∅` 292, `t`→`∅` 244, `r`→`∅` 225, `i`→`∅` 204, `s`→`∅` 195, `d`→`∅` 180, `a`→`∅` 177, `l`→`∅` 138
- **kraken-bern-hertwig-v2** top confusions (ref→hyp): ` `→`∅` 454, `e`→`∅` 327, `n`→`∅` 245, `t`→`∅` 223, `r`→`∅` 207, `i`→`∅` 181, `a`→`∅` 160, `s`→`∅` 159, `d`→`∅` 153, `u`→`∅` 125
- **kraken-bern-schlussredaktion-v1** top confusions (ref→hyp): ` `→`∅` 500, `e`→`∅` 341, `n`→`∅` 253, `t`→`∅` 239, `r`→`∅` 212, `i`→`∅` 181, `s`→`∅` 177, `a`→`∅` 162, `d`→`∅` 153, `u`→`∅` 126
- **kraken-bern-combined-v1** top confusions (ref→hyp): ` `→`∅` 1146, `e`→`∅` 773, `n`→`∅` 626, `t`→`∅` 468, `r`→`∅` 441, `i`→`∅` 405, `s`→`∅` 392, `a`→`∅` 322, `d`→`∅` 279, `h`→`∅` 251

## bern-hertwig-train-300 — 300 lines, list sha256 `bd4e5a8a4957ac59d2c9126adae1c2006d860543f2b4a9fb94cbc5953a7fe25e`

| model | CER | WER | collapsed | length ratio | blank share | params | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kraken-cursive_inzigkofen | 0.7059 | 1.2603 | 23 | 0.963 | 0.699 | 4,097,926 | 17.5 |
| kraken-bern-hertwig-v1 | 0.1765 | 0.4836 | 19 | 0.915 | 0.751 | 4,099,931 | 15.3 |
| kraken-bern-hertwig-v2 | 0.0828 | 0.2054 | 12 | 0.951 | 0.722 | 4,099,931 | 15.5 |
| kraken-bern-schlussredaktion-v1 | 0.2138 | 0.6310 | 17 | 0.895 | 0.742 | 4,100,733 | 15.0 |
| kraken-bern-combined-v1 | 0.1853 | 0.4848 | 26 | 0.909 | 0.741 | 4,101,134 | 15.6 |

- **kraken-cursive_inzigkofen** top confusions (ref→hyp): `∅`→` ` 165, ` `→`∅` 121, `s`→`ſ` 118, `e`→`∅` 107, `∅`→`i` 90, `n`→`∅` 88, `∅`→`e` 78, `s`→`∅` 78, `e`→` ` 73, `r`→`∅` 66
- **kraken-bern-hertwig-v1** top confusions (ref→hyp): ` `→`∅` 168, `n`→`∅` 67, `/`→`∅` 61, `e`→`∅` 55, `∅`→` ` 54, `,`→`/` 41, `t`→`∅` 40, `i`→`∅` 36, `r`→`∅` 35, `s`→`∅` 35
- **kraken-bern-hertwig-v2** top confusions (ref→hyp): ` `→`∅` 75, `n`→`∅` 38, `e`→`∅` 36, `s`→`∅` 26, `t`→`∅` 25, `i`→`∅` 24, `r`→`∅` 22, `∅`→` ` 21, `h`→`∅` 21, `d`→`∅` 20
- **kraken-bern-schlussredaktion-v1** top confusions (ref→hyp): ` `→`∅` 263, `/`→`∅` 85, `/`→`,` 68, `n`→`∅` 66, `e`→`∅` 66, `d`→`D` 50, `u`→`v` 47, `i`→`∅` 43, `r`→`∅` 42, `s`→`∅` 38
- **kraken-bern-combined-v1** top confusions (ref→hyp): ` `→`∅` 163, `n`→`∅` 69, `e`→`∅` 56, `/`→`∅` 54, `∅`→` ` 53, `t`→`∅` 51, `i`→`∅` 49, `r`→`∅` 46, `s`→`∅` 44, `c`→`∅` 35

## blb-train-300 — 300 lines, list sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`

| model | CER | WER | collapsed | length ratio | blank share | params | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kraken-german-xix-v1 | 0.3039 | 0.7337 | 23 | 0.867 | 0.776 | 4,083,089 | 25.7 |
| kraken-german-xix-blb-v1 | 0.0646 | 0.1711 | 10 | 0.969 | 0.740 | 4,111,961 | 22.9 |

- **kraken-german-xix-v1** top confusions (ref→hyp): ` `→`∅` 525, `e`→`∅` 167, `i`→`∅` 103, `r`→`∅` 91, `n`→`∅` 88, `t`→`∅` 77, `s`→`∅` 75, `c`→`∅` 55, `a`→`∅` 51, `.`→`∅` 49
- **kraken-german-xix-blb-v1** top confusions (ref→hyp): ` `→`∅` 103, `e`→`∅` 40, `n`→`∅` 35, `i`→`∅` 22, `r`→`∅` 21, `s`→`∅` 18, `∅`→` ` 18, `.`→`∅` 17, `o`→`∅` 13, `a`→`∅` 12

## medieval-train-300 — 300 lines, list sha256 `a2160d4ef170a31e7ab0f3b96d631272347e281a1cbc7f65423f5d0dd8f5dd3d`

| model | CER | WER | collapsed | length ratio | blank share | params | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kraken-medieval-shard00-std | 0.6301 | 0.9583 | 159 | 0.530 | 0.865 | 4,033,766 | 28.7 |
| kraken-plus-shard00-cr1 | 0.6378 | 0.9556 | 132 | 0.573 | 0.730 | 4,148,545 | 11.9 |
| kraken-medieval-shard00-v1 | 0.6239 | 0.9635 | 123 | 0.603 | 0.696 | 4,148,494 | 11.9 |
| kraken-cursive_inzigkofen | 0.7375 | 1.2157 | 31 | 0.923 | 0.740 | 4,097,926 | 26.4 |
| kraken-collapse-probe-131-cr255 | 0.9846 | 0.9996 | 298 | 0.019 | 0.647 | 15,207,871 | 16.0 |

- **kraken-medieval-shard00-std** top confusions (ref→hyp): ` `→`∅` 801, `e`→`∅` 676, `n`→`∅` 509, `i`→`∅` 410, `r`→`∅` 369, `t`→`∅` 315, `s`→`∅` 307, `u`→`∅` 305, `a`→`∅` 257, `d`→`∅` 244
- **kraken-plus-shard00-cr1** top confusions (ref→hyp): ` `→`∅` 706, `e`→`∅` 578, `n`→`∅` 433, `i`→`∅` 390, `r`→`∅` 335, `t`→`∅` 298, `s`→`∅` 295, `u`→`∅` 275, `a`→`∅` 242, `d`→`∅` 218
- **kraken-medieval-shard00-v1** top confusions (ref→hyp): ` `→`∅` 589, `e`→`∅` 583, `n`→`∅` 426, `i`→`∅` 388, `r`→`∅` 306, `u`→`∅` 261, `t`→`∅` 258, `s`→`∅` 245, `a`→`∅` 228, `d`→`∅` 201
- **kraken-cursive_inzigkofen** top confusions (ref→hyp): `e`→`∅` 189, ` `→`∅` 177, `∅`→` ` 177, `n`→`∅` 159, `s`→`∅` 122, `r`→`∅` 122, `a`→`∅` 122, `∅`→`i` 120, `∅`→`e` 115, `e`→`i` 114
- **kraken-collapse-probe-131-cr255** top confusions (ref→hyp): ` `→`∅` 1975, `e`→`∅` 1439, `n`→`∅` 1076, `i`→`∅` 809, `t`→`∅` 723, `r`→`∅` 688, `a`→`∅` 672, `s`→`∅` 628, `u`→`∅` 563, `o`→`∅` 441

## medieval-val-300-archive — 300 lines, list sha256 `7e3f6eebc08e717e444e3f644b38075152a4abe71550723fad12bba34b9b1290`

| model | CER | WER | collapsed | length ratio | blank share | params | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kraken-medieval-shard00-std | 0.6703 | 0.9625 | 181 | 0.450 | 0.010 | 4,033,766 | 28.9 |
| kraken-plus-shard00-cr1 | 0.7020 | 0.9701 | 161 | 0.462 | 0.010 | 4,148,545 | 12.1 |

- **kraken-medieval-shard00-std** top confusions (ref→hyp): ` `→`∅` 952, `e`→`∅` 776, `n`→`∅` 496, `i`→`∅` 493, `u`→`∅` 405, `s`→`∅` 395, `t`→`∅` 385, `r`→`∅` 368, `a`→`∅` 348, `l`→`∅` 307
- **kraken-plus-shard00-cr1** top confusions (ref→hyp): ` `→`∅` 981, `e`→`∅` 742, `n`→`∅` 500, `i`→`∅` 488, `s`→`∅` 400, `u`→`∅` 390, `t`→`∅` 382, `r`→`∅` 371, `a`→`∅` 353, `d`→`∅` 286

## medieval-val-300 — 300 lines, list sha256 `a2f0f40006e7a377b92cef13ad0274ed5fcf91691c545a86f4309767775e061f`

| model | CER | WER | collapsed | length ratio | blank share | params | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kraken-plus-shard00-cr1 | 0.7213 | 0.9745 | 167 | 0.443 | 0.792 | 4,148,545 | 15.6 |
| kraken-medieval-shard00-std | 0.6744 | 0.9597 | 183 | 0.453 | 0.885 | 4,033,766 | 28.8 |
| kraken-medieval-shard00-v1 | 0.7080 | 0.9792 | 156 | 0.474 | 0.753 | 4,148,494 | 12.6 |
| kraken-collapse-probe-131-cr255 | 0.9900 | 0.9987 | 300 | 0.013 | 0.663 | 15,207,871 | 20.1 |
| kraken-collapse-probe-131-cr1 | 0.9900 | 0.9987 | 300 | 0.013 | 0.663 | 4,153,791 | 12.8 |
| kraken-cursive_inzigkofen | 0.7388 | 1.2436 | 34 | 0.902 | 0.760 | 4,097,926 | 27.0 |
| kraken-german-xix-blb-v1 | 0.5454 | 0.9826 | 59 | 0.774 | 0.812 | 4,111,961 | 26.3 |

- **kraken-plus-shard00-cr1** top confusions (ref→hyp): ` `→`∅` 1058, `e`→`∅` 760, `i`→`∅` 529, `n`→`∅` 502, `t`→`∅` 412, `s`→`∅` 399, `u`→`∅` 389, `a`→`∅` 377, `r`→`∅` 376, `l`→`∅` 281
- **kraken-medieval-shard00-std** top confusions (ref→hyp): ` `→`∅` 967, `e`→`∅` 743, `i`→`∅` 496, `n`→`∅` 492, `s`→`∅` 384, `t`→`∅` 384, `u`→`∅` 383, `a`→`∅` 370, `r`→`∅` 363, `l`→`∅` 289
- **kraken-medieval-shard00-v1** top confusions (ref→hyp): ` `→`∅` 903, `e`→`∅` 758, `n`→`∅` 501, `i`→`∅` 498, `s`→`∅` 381, `a`→`∅` 378, `r`→`∅` 370, `u`→`∅` 363, `t`→`∅` 349, `l`→`∅` 286
- **kraken-collapse-probe-131-cr255** top confusions (ref→hyp): ` `→`∅` 2056, `e`→`∅` 1553, `n`→`∅` 955, `i`→`∅` 821, `t`→`∅` 720, `a`→`∅` 719, `s`→`∅` 664, `r`→`∅` 629, `u`→`∅` 613, `l`→`∅` 508
- **kraken-collapse-probe-131-cr1** top confusions (ref→hyp): ` `→`∅` 2056, `e`→`∅` 1553, `n`→`∅` 955, `i`→`∅` 821, `t`→`∅` 720, `a`→`∅` 719, `s`→`∅` 664, `r`→`∅` 629, `u`→`∅` 613, `l`→`∅` 508
- **kraken-cursive_inzigkofen** top confusions (ref→hyp): ` `→`∅` 218, `e`→`∅` 207, `∅`→` ` 160, `s`→`∅` 146, `a`→`∅` 129, `n`→`∅` 128, `i`→`∅` 127, `r`→`∅` 126, `t`→`∅` 125, `e`→`i` 116
- **kraken-german-xix-blb-v1** top confusions (ref→hyp): ` `→`∅` 366, `e`→`∅` 357, `i`→`∅` 216, `n`→`∅` 213, `t`→`∅` 192, `r`→`∅` 185, `u`→`∅` 162, `a`→`∅` 156, `s`→`∅` 150, `c`→`∅` 117
