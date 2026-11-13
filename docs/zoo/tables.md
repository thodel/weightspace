### T1. Models, one row each (zoo, then in-house)

Group = architecture group of T2. `blank raw` = bias of the CTC blank row; `blank rel` = that bias minus the mean of the model's other biases (the softmax is unchanged when a constant is added to all biases, so only the relative value can matter); `rank` = place of the blank by bias among all output rows (1 = highest); `Lat` = share of the codec that is Latin letters.

| model | DOI | grp | params | codec | Lat | comb. marks | blank raw | blank rel | rank | blank ‖row‖ / mean other |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `austrian_fraktur` | 7933402 | A | 4,067,851 | 186 | 0.46 | 0 | +0.14 | +1.72 | 1 of 187 | 0.19 |
| `bastarda_inzigkofen` | 18207779 | A | 4,096,322 | 257 | 0.35 | 42 | +0.78 | +1.37 | 1 of 258 | 0.32 |
| `bifrost_old_norse` | 15366732 | A | 4,097,525 | 260 | 0.41 | 44 | +0.34 | +0.56 | 1 of 261 | 0.39 |
| `bohemian_19th` | 11673242 | A | 4,047,400 | 135 | 0.56 | 0 | +0.70 | +1.73 | 7 of 136 | 0.33 |
| `catmus_gothic_print` | 10599911 | B | 5,716,595 | 210 | 0.38 | 31 | +0.61 | +1.80 | 2 of 211 | 0.16 |
| `catmus_medieval` | 15030337 | A | 4,093,916 | 251 | 0.35 | 40 | +0.57 | +0.64 | 1 of 252 | 0.35 |
| `catmus_print_large` | 10592716 | B | 5,721,725 | 220 | 0.32 | 16 | +1.13 | +1.13 | 1 of 221 | 0.21 |
| `cremma_medieval` | 7631619 | B | 5,696,588 | 171 | 0.38 | 27 | +1.20 | +1.11 | 1 of 172 | 0.17 |
| `cursive_inzigkofen` | 18207767 | A | 4,097,926 | 261 | 0.35 | 43 | +0.82 | +1.48 | 1 of 262 | 0.28 |
| `cyrillic_uncial` | 7755483 | A | 4,082,287 | 222 | 0.08 | 61 | +1.47 | +1.88 | 1 of 223 | 0.41 |
| `english_print` | 2577813 | E | 732,906 | 89 | 0.58 | 0 | -0.69 | -0.67 | 83 of 90 | 0.17 |
| `estournelles_typewritten` | 10556673 | F | 4,352,010 | 105 | 0.70 | 0 | -1.04 | -1.10 | 106 of 106 | 0.30 |
| `fondue_gd_v2` | 21536798 | A | 4,079,881 | 216 | 0.31 | 16 | +1.88 | +2.34 | 1 of 217 | 0.25 |
| `french_vietnamese` | 17690418 | B | 5,731,985 | 240 | 0.30 | 20 | +1.12 | +1.24 | 1 of 241 | 0.21 |
| `gallicorpora` | 7410529 | A | 4,057,425 | 160 | 0.39 | 17 | +0.10 | +0.13 | 2 of 161 | 0.75 |
| `german_print` | 10519596 | A | 4,106,748 | 283 | 0.38 | 4 | +0.41 | +0.83 | 15 of 284 | 0.16 |
| `glagolitic_print` | 7755504 | A | 4,055,019 | 154 | 0.42 | 14 | +0.40 | +0.90 | 1 of 155 | 0.60 |
| `hebrew_sephardi` | 5468665 | A | 4,021,335 | 70 | 0.06 | 14 | +0.15 | +0.82 | 1 of 71 | 0.49 |
| `kuzushiji` | 13942714 | A | 10,772,571 | 16,906 | 0.00 | 0 | +2.00 | +5.92 | 1 of 16907 | 0.53 |
| `latin_expanded_11c` | 13736584 | A | 4,023,340 | 75 | 0.73 | 4 | +0.04 | +0.10 | 7 of 76 | 0.61 |
| `latin_graphematic_11c` | 13741957 | A | 4,037,375 | 110 | 0.61 | 7 | +0.06 | +0.11 | 13 of 111 | 0.51 |
| `latin_incunabula` | 11113737 | B | 5,723,264 | 223 | 0.37 | 11 | +1.25 | +1.44 | 1 of 224 | 0.25 |
| `lectaurep_french` | 6542744 | A | 4,033,766 | 101 | 0.54 | 6 | +0.37 | +0.62 | 4 of 102 | 0.31 |
| `manu_mcfondue` | 10886224 | A | 4,096,723 | 258 | 0.44 | 5 | +1.19 | +2.00 | 14 of 259 | 0.25 |
| `mccatmus` | 13788177 | A | 4,039,781 | 116 | 0.49 | 11 | +0.61 | +0.77 | 1 of 117 | 0.59 |
| `medieval_latin_french_abbreviated` | 7516310 | B | 5,672,990 | 125 | 0.34 | 23 | +1.00 | +1.00 | 2 of 126 | 0.16 |
| `medieval_latin_french_expanded` | 7516057 | A | 4,037,375 | 110 | 0.35 | 22 | +0.13 | +0.19 | 1 of 111 | 0.57 |
| `mendelssohn_letters` | 18207676 | A | 4,111,560 | 295 | 0.41 | 7 | +0.14 | +8.89 | 1 of 296 | 0.25 |
| `old_norse_am305` | 20529753 | A | 4,108,352 | 287 | 0.37 | 31 | +0.70 | +0.98 | 1 of 288 | 0.60 |
| `openiti_arabic_print` | 7050270 | A | 4,071,861 | 196 | 0.24 | 20 | +0.53 | +0.53 | 16 of 197 | 0.17 |
| `openiti_ottoman_print` | 7050342 | A | 4,071,861 | 196 | 0.24 | 20 | +0.76 | +0.86 | 11 of 197 | 0.11 |
| `openiti_persian_print` | 7051644 | A | 4,071,861 | 196 | 0.24 | 20 | +0.80 | +0.74 | 25 of 197 | 0.10 |
| `peraire_french` | 8193498 | A | 4,041,385 | 120 | 0.68 | 0 | +0.75 | +0.92 | 1 of 121 | 0.52 |
| `prima` | 18220238 | A | 4,051,811 | 146 | 0.52 | 14 | +0.65 | +0.85 | 1 of 147 | 0.64 |
| `samaritan` | 13814200 | A | 4,008,904 | 39 | 0.00 | 0 | +0.71 | +1.27 | 1 of 40 | 0.27 |
| `textualis_inzigkofen` | 18207719 | C | 6,269,400 | 279 | 0.49 | 25 | +0.36 | +7.47 | 32 of 280 | 0.35 |
| `tridis_v2` | 13862096 | C | 6,266,835 | 274 | 0.50 | 21 | +0.21 | +6.84 | 61 of 275 | 0.37 |
| `bern-combined-v1` | — | A | 4,101,134 | 269 | 0.35 | 45 | +1.00 | +2.58 | 1 of 270 | 0.20 |
| `bern-hertwig-v1` | — | A | 4,099,931 | 266 | 0.35 | 44 | +0.88 | +1.80 | 1 of 267 | 0.25 |
| `bern-hertwig-v2` | — | A | 4,099,931 | 266 | 0.35 | 44 | +0.88 | +2.00 | 1 of 267 | 0.23 |
| `bern-schlussredaktion-v1` | — | A | 4,100,733 | 268 | 0.34 | 45 | +0.94 | +2.39 | 1 of 269 | 0.20 |
| `collapse-probe-131-cr1` | — | D | 4,153,791 | 162 | 0.45 | 18 | +0.01 | +0.02 | 16 of 163 | 0.92 |
| `german-xix-blb-v1` | — | A | 4,111,961 | 296 | 0.32 | 26 | +1.23 | +3.15 | 1 of 297 | 0.31 |
| `german-xix-v1` | — | A | 4,083,089 | 224 | 0.36 | 15 | +1.12 | +2.99 | 1 of 225 | 0.26 |
| `medieval-shard00-std` | — | A | 4,033,766 | 101 | 0.57 | 9 | +0.13 | +0.33 | 2 of 102 | 0.56 |
| `medieval-shard00-v1` | — | G | 4,148,494 | 101 | 0.57 | 9 | +0.37 | +0.66 | 1 of 102 | 0.58 |
| `plus-shard00-cr1` | — | D | 4,148,545 | 101 | 0.57 | 9 | +0.03 | +0.09 | 32 of 102 | 0.53 |

### T2. Architecture groups (VGSL with block names, dropout rates and the output width removed)

| grp | specification | zoo | in-house | members (zoo) |
|---|---|---:|---:|---|
| A | `[1,120,0,1 Cr3,13,32 Do Mp2,2 Cr3,13,32 Do Mp2,2 Cr3,9,64 Do Mp2,2 Cr3,9,64 Do S1(1x0)1,3 Lbx200 Do Lbx200 Do Lbx200 Do O1c<N>]` | 27 | 7 | austrian_fraktur, bastarda_inzigkofen, bifrost_old_norse, bohemian_19th, catmus_medieval, cursive_inzigkofen, cyrillic_uncial, fondue_gd_v2, gallicorpora, german_print, glagolitic_print, hebrew_sephardi, kuzushiji, latin_expanded_11c, latin_graphematic_11c, lectaurep_french, manu_mcfondue, mccatmus, medieval_latin_french_expanded, mendelssohn_letters, old_norse_am305, openiti_arabic_print, openiti_ottoman_print, openiti_persian_print, peraire_french, prima, samaritan |
| B | `[1,120,0,1 Cr4,2,32,4,2 Gn32 Cr4,2,64,1,1 Gn32 Mp4,2,4,2 Cr3,3,128,1,1 Gn32 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do Lbx256 Do Lbx256 Do O1c<N>]` | 6 | 0 | catmus_gothic_print, catmus_print_large, cremma_medieval, french_vietnamese, latin_incunabula, medieval_latin_french_abbreviated |
| C | `[1,140,0,1 Cr4,16,32 Do Mp2,2 Cr4,16,32 Do Mp2,2 Cr3,8,64 Do Mp2,2 Cr3,8,64 Do S1(1x0)1,3 Lbx256 Do Lbx256 Do Lbx256 Do O1c<N>]` | 2 | 0 | textualis_inzigkofen, tridis_v2 |
| D | `[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do Lbx256 Do Lbx256 Do Cr1,1,85,1,1 O1c<N>]` | 0 | 2 | — |
| E | `[1,48,0,1 Cr3,3,32 Do Mp2,2 Cr3,3,64 Do Mp2,2 S1(1x12)1,3 Lbx100 Do O1c<N>]` | 1 | 0 | english_print |
| F | `[1,48,0,1 Cr4,2,32,4,2 Gn32 Cr4,2,64,1,1 Gn32 Mp4,2,4,2 Cr3,3,128,1,1 Gn32 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do Lbx256 Do Lbx256 Do O1c<N>]` | 1 | 0 | estournelles_typewritten |
| G | `[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do Lbx256 Do Lbx256 Do O1c<N>]` | 0 | 1 | — |

Within group A the specification strings differ only in how the dropout layers are written: {'dropout written `Do.…` form': 11, 'dropout written `Do0.1,2` form': 16}. Dropout has no parameters, so these are one architecture.

### T3. Documented parent → child pairs and what the weights say

Distance = ‖a − b‖ / mean(‖a‖, ‖b‖) over all non-output weights (symmetric). Two models whose weights are unrelated sit near √2 = 1.41; the median over all pairs of group A is 1.42.

| child | documented parent | source of the record | same architecture | distance | rank of parent among child's neighbours | verdict |
|---|---|---|---|---:|---:|---|
| `bastarda_inzigkofen` | `catmus_medieval` | gateway base_model 10.5281/zenodo.15030337; README: 'obtained by fine-tuning the CATMuS Medieval model' | yes | 0.72 | 1 of 33 | weights agree with the record |
| `cursive_inzigkofen` | `catmus_medieval` | gateway base_model; README: 'obtained by fine-tuning the CATMuS Medieval model' | yes | 0.74 | 5 of 33 | weights agree with the record |
| `bifrost_old_norse` | `catmus_medieval` | gateway base_model 10.5281/zenodo.15030337; the Zenodo text read has no base sentence | yes | 1.36 | 8 of 33 | **weights do not support the record**: as far from the documented parent as from unrelated models |
| `textualis_inzigkofen` | `tridis_v2` | gateway base_model 10.5281/zenodo.13862096; README: 'obtained by fine-tuning the TriDiS v2 model' | yes | 0.33 | 1 of 1 | weights agree with the record |
| `catmus_gothic_print` | `catmus_medieval` | gateway base_model; README: 'fine-tuned on CATMuS Medieval model' | no | — | — | architecture differs: not a weight-level fine-tune of this file: child `[1,120,0,1 Cr4,2,32,4,2 Gn32 Cr4,2…`, parent `[1,120,0,1 Cr3,13,32 Do Mp2,2 Cr3,…` |
| `french_vietnamese` | `catmus_print_large` | gateway base_model 10.5281/zenodo.10592716; README: 'transfer learning from the CATMuS-Print base model' | yes | 0.19 | 1 of 5 | weights agree with the record |
| `latin_incunabula` | `catmus_print_large` | gateway base_model 10.5281/zenodo.10592716; the README sentences read name no base | yes | 0.33 | 1 of 5 | weights agree with the record |
| `openiti_persian_print` | `openiti_arabic_print` | gateway base_model 10.5281/zenodo.7050270; Zenodo title 'Printed Persian Base Model Trained on the OpenITI Corpus' | yes | 0.70 | 2 of 33 | weights agree with the record |
| `openiti_ottoman_print` | `openiti_arabic_print` | gateway base_model 10.5281/zenodo.7050270; Zenodo title 'Printed Ottoman Base Model Trained on the OpenITI Corpus' | yes | 0.72 | 2 of 33 | weights agree with the record |
| `german-xix-v1` | `bohemian_19th` | trainer record: job 20261007T131819Z-kraken-german-xix-v1, job.json request.base_model = kraken-bohemian_19th (resize union, lrate 1e-4, cosine, 97 epochs run); first suggested by the weights, then confirmed there | yes | 0.70 | 2 of 33 | weights agree with the record |
| `bern-hertwig-v1` | `cursive_inzigkofen` | docs/MODELS.md (trainer record) | yes | 0.13 | 1 of 33 | weights agree with the record |
| `bern-hertwig-v2` | `cursive_inzigkofen` | docs/MODELS.md (trainer record) | yes | 0.28 | 4 of 33 | weights agree with the record |
| `bern-hertwig-v2` | `bern-hertwig-v1` | docs/READINGS.md (read as a continuation of v1; docs/MODELS.md lists the same base cursive_inzigkofen for both) | yes | 0.23 | 2 of 33 | weights agree with the record |
| `bern-schlussredaktion-v1` | `bern-hertwig-v2` | docs/MODELS.md (trainer record) | yes | 0.20 | 1 of 33 | weights agree with the record |
| `bern-combined-v1` | `bern-hertwig-v2` | docs/MODELS.md (trainer record) | yes | 0.23 | 2 of 33 | weights agree with the record |
| `german-xix-blb-v1` | `german-xix-v1` | docs/MODELS.md (trainer record) | yes | 0.27 | 1 of 33 | weights agree with the record |

Documented in text, but the parent file is not among those surveyed (so nothing is tested):

- `kuzushiji`: Chat model by Colin Brisson (Zenodo), named in the README: 'finetuned on top of the Chat model'
- `mendelssohn_letters`: 'HTR model for German manuscripts trained from several datasets' (eScriptorium base); not identified with a DOI in the text read
- `old_norse_am305`: 'fine-tuned from a base model trained on AM 302 fol (Grágás)'; that model is not in the gateway list
- `peraire_french`: Manu McFrench (Zenodo 6657809), named in the metadata; not in the gateway list

### T4. Relations that only the weights suggest (not documented in any text read)

| child | parent (direction from the codec) | distance | codec | shared characters at the same row index |
|---|---|---:|---|---:|
| `prima` | `mccatmus` | 0.26 | 116 → 146 (parent's codec is a subset: True) | 116 of 116 |
| `glagolitic_print` | `hebrew_sephardi` | 0.83 | 70 → 154 (parent's codec is a subset: True) | 70 of 70 |

Other pairs below 1.2 in group A that no record explains (distance, no direction): `openiti_ottoman_print` – `openiti_persian_print` 0.28; `hebrew_sephardi` – `samaritan` 0.95; `glagolitic_print` – `samaritan` 1.02.

### T5. Observation 1 — rows of rare characters (proxy: characters outside basic Latin letters, digits, space and ASCII punctuation)

`special` = such rows among the characters both codecs share. Frequent = the ten letters `e n i a r t s o u d`. Medians are over rows. A pair *supports* the observation when all four hold: (a) most special rows have a larger norm in the child, (b) most special rows have a lower bias, (c) the special rows moved more than the frequent ones (median relative L2), (d) the top decile of rows by relative move contains at least its share of special rows.

| pair | special rows | norm up | bias down | median norm ratio (special / frequent) | median bias Δ (special / frequent) | median rel. move (special / frequent) | top decile special / share | a b c d |
|---|---:|---:|---:|---|---|---|---|---|
| `catmus_medieval` → `bastarda_inzigkofen` | 178 | 178 | 174 | 1.41 / 1.40 | -0.71 / -0.06 | 0.72 / 0.69 | 21 of 25 / 0.71 | ✓✓✓✓ |
| `catmus_medieval` → `cursive_inzigkofen` | 178 | 177 | 167 | 1.52 / 1.35 | -0.84 / -0.06 | 0.82 / 0.66 | 15 of 25 / 0.71 | ✓✓✓✗ |
| `tridis_v2` → `textualis_inzigkofen` | 199 | 125 | 186 | 1.02 / 1.06 | -0.80 / +0.04 | 0.10 / 0.17 | 14 of 27 / 0.73 | ✓✓✗✗ |
| `catmus_print_large` → `french_vietnamese` | 145 | 142 | 133 | 1.09 / 1.19 | -0.20 / -0.02 | 0.17 / 0.38 | 10 of 22 / 0.66 | ✓✓✗✗ |
| `catmus_print_large` → `latin_incunabula` | 51 | 15 | 32 | 0.85 / 1.17 | -0.06 / -0.02 | 0.71 / 0.45 | 6 of 12 / 0.42 | ✗✓✓✓ |
| `openiti_arabic_print` → `openiti_persian_print` | 125 | 125 | 74 | 1.52 / 1.64 | -0.06 / +0.14 | 0.63 / 0.73 | 6 of 19 / 0.64 | ✓✓✗✗ |
| `openiti_arabic_print` → `openiti_ottoman_print` | 125 | 125 | 78 | 1.56 / 1.72 | -0.07 / -0.31 | 0.70 / 0.83 | 8 of 19 / 0.64 | ✓✓✗✗ |
| `bohemian_19th` → `german-xix-v1` | 61 | 46 | 45 | 1.06 / 0.88 | -2.02 / +0.05 | 0.54 / 0.39 | 13 of 13 / 0.45 | ✓✓✓✓ |
| `cursive_inzigkofen` → `bern-hertwig-v1` | 188 | 181 | 182 | 1.15 / 0.97 | -0.41 / +0.03 | 0.27 / 0.16 | 26 of 26 / 0.72 | ✓✓✓✓ |
| `cursive_inzigkofen` → `bern-hertwig-v2` | 188 | 182 | 179 | 1.23 / 1.03 | -0.72 / +0.01 | 0.40 / 0.26 | 26 of 26 / 0.72 | ✓✓✓✓ |
| `bern-hertwig-v1` → `bern-hertwig-v2` | 192 | 189 | 180 | 1.07 / 1.06 | -0.31 / -0.01 | 0.14 / 0.18 | 20 of 26 / 0.72 | ✓✓✗✓ |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | 192 | 172 | 176 | 1.07 / 0.96 | -0.53 / -0.00 | 0.12 / 0.17 | 22 of 26 / 0.72 | ✓✓✗✓ |
| `bern-hertwig-v2` → `bern-combined-v1` | 192 | 173 | 175 | 1.08 / 0.95 | -0.74 / -0.03 | 0.14 / 0.21 | 23 of 26 / 0.72 | ✓✓✗✓ |
| `german-xix-v1` → `german-xix-blb-v1` | 149 | 110 | 108 | 1.01 / 1.02 | -0.69 / +0.06 | 0.18 / 0.15 | 21 of 22 / 0.67 | ✓✓✓✓ |
| `mccatmus` → `prima` (inferred) | 41 | 40 | 35 | 1.15 / 0.95 | -0.24 / +0.01 | 0.37 / 0.21 | 6 of 11 / 0.35 | ✓✓✓✓ |
| `hebrew_sephardi` → `glagolitic_print` (inferred) | 56 | 55 | 55 | 1.54 / – | -0.33 / – | 1.23 / – | 7 of 7 / 0.80 | ✓✓–✓ |

Tally over the 14 documented pairs with related weights: (a) 13, (b) 14, (c) 7, (d) 9; all four in 5. Zoo pairs only (7): all four in 1.

The companion claim — the ten frequent letters have the smallest row norms — can be read on every model without a pair: mean norm of the ten frequent rows over the mean of all other rows is below 1 in 40 of 43 models (median 0.56; zoo only 30 of 33, median 0.61); at least five of the ten weakest non-blank rows are frequent letters in 29 of 43. 4 models (non-Latin codecs) lack one of the ten letters and are not counted.

### T6. Observation 2 — the blank bias

Δ raw = change of `bias[blank]`; Δ rel = change of (`bias[blank]` − mean of the other biases); `Δ mean other` = change of the mean of the other biases.

| pair | Δ raw | Δ mean other | Δ rel | blank row, relative L2 |
|---|---:|---:|---:|---:|
| `catmus_medieval` → `bastarda_inzigkofen` | +0.210 | -0.523 | +0.734 | 0.52 |
| `catmus_medieval` → `cursive_inzigkofen` | +0.241 | -0.604 | +0.845 | 0.50 |
| `tridis_v2` → `textualis_inzigkofen` | +0.151 | -0.478 | +0.629 | 0.21 |
| `catmus_print_large` → `french_vietnamese` | -0.009 | -0.114 | +0.105 | 0.34 |
| `catmus_print_large` → `latin_incunabula` | +0.116 | -0.190 | +0.305 | 0.52 |
| `openiti_arabic_print` → `openiti_persian_print` | +0.263 | +0.053 | +0.210 | 0.64 |
| `openiti_arabic_print` → `openiti_ottoman_print` | +0.228 | -0.100 | +0.328 | 0.78 |
| `bohemian_19th` → `german-xix-v1` | +0.419 | -0.842 | +1.261 | 0.62 |
| `cursive_inzigkofen` → `bern-hertwig-v1` | +0.062 | -0.258 | +0.320 | 0.18 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | +0.064 | -0.456 | +0.520 | 0.29 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | +0.002 | -0.198 | +0.200 | 0.21 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | +0.056 | -0.335 | +0.391 | 0.20 |
| `bern-hertwig-v2` → `bern-combined-v1` | +0.118 | -0.456 | +0.574 | 0.25 |
| `german-xix-v1` → `german-xix-blb-v1` | +0.113 | -0.050 | +0.163 | 0.23 |
| `mccatmus` → `prima` (inferred) | +0.036 | -0.048 | +0.084 | 0.17 |
| `hebrew_sephardi` → `glagolitic_print` (inferred) | +0.253 | +0.174 | +0.079 | 0.74 |

Tally over the 14 documented related pairs: Δ raw > 0 in 13, Δ rel > 0 in 14; the mean of the other biases fell in 13.

Without pairs, over all 47 models: the blank has the highest bias in 28, blank bias − mean of the others is positive in 45 (raw bias positive in 45), and the blank's row norm is below the mean of the other rows in 47. Zoo only: 21 of 37 with the highest bias, relative value median 0.98; in-house 10: median 1.90.

### T7. Observation 3 — which stack moves

Pooled relative L2 over the **weight** tensors of a stack (biases excluded: their norms are tiny and their ratios large). `READINGS def.` = the definition of docs/READINGS.md: per-tensor relative L2 averaged per block, biases included, the all-zero `bias_ih` of the LSTMs counted as 0.

| pair | conv | BiLSTM | output rows | C_0 / C_3 / C_6 / C_9 (weights) | READINGS def.: conv / BiLSTM |
|---|---:|---:|---:|---|---|
| `catmus_medieval` → `bastarda_inzigkofen` | 0.895 | 0.880 | 0.696 | 0.26 / 0.55 / 0.93 / 1.08 | 0.623 / 0.452 |
| `catmus_medieval` → `cursive_inzigkofen` | 0.900 | 0.898 | 0.809 | 0.24 / 0.57 / 0.93 / 1.09 | 0.608 / 0.463 |
| `tridis_v2` → `textualis_inzigkofen` | 0.393 | 0.340 | 0.132 | 0.12 / 0.17 / 0.43 / 0.52 | 1.592 / 0.202 |
| `catmus_print_large` → `french_vietnamese` | 0.129 | 0.195 | 0.200 | 0.05 / 0.12 / 0.13 | 0.103 / 0.106 |
| `catmus_print_large` → `latin_incunabula` | 0.258 | 0.339 | 0.632 | 0.07 / 0.22 / 0.27 | 0.156 / 0.186 |
| `openiti_arabic_print` → `openiti_persian_print` | 0.755 | 0.828 | 0.700 | 0.66 / 0.65 / 0.71 / 0.81 | 0.852 / 0.491 |
| `openiti_arabic_print` → `openiti_ottoman_print` | 0.781 | 0.858 | 0.762 | 0.66 / 0.66 / 0.74 / 0.84 | 0.944 / 0.513 |
| `bohemian_19th` → `german-xix-v1` | 0.712 | 0.723 | 0.509 | 1.29 / 0.59 / 0.72 / 0.74 | 0.731 / 0.401 |
| `cursive_inzigkofen` → `bern-hertwig-v1` | 0.117 | 0.128 | 0.265 | 0.07 / 0.09 / 0.13 / 0.12 | 0.102 / 0.075 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | 0.259 | 0.294 | 0.391 | 0.13 / 0.21 / 0.27 / 0.27 | 0.204 / 0.162 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | 0.204 | 0.239 | 0.145 | 0.10 / 0.17 / 0.21 / 0.22 | 0.151 / 0.127 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | 0.183 | 0.205 | 0.156 | 0.08 / 0.14 / 0.19 / 0.20 | 0.130 / 0.110 |
| `bern-hertwig-v2` → `bern-combined-v1` | 0.210 | 0.241 | 0.175 | 0.09 / 0.16 / 0.21 / 0.23 | 0.145 / 0.129 |
| `german-xix-v1` → `german-xix-blb-v1` | 0.260 | 0.270 | 0.204 | 0.12 / 0.17 / 0.26 / 0.29 | 0.333 / 0.139 |
| `mccatmus` → `prima` (inferred) | 0.233 | 0.281 | 0.342 | 0.11 / 0.19 / 0.23 / 0.25 | 0.182 / 0.140 |
| `hebrew_sephardi` → `glagolitic_print` (inferred) | 0.722 | 1.157 | 1.004 | 0.29 / 0.58 / 0.71 / 0.79 | 0.586 / 0.588 |

Tally over the 14 documented related pairs: conv < BiLSTM in 11; conv < output rows in 4; the first conv block moves least of the conv blocks in 12; conv blocks increase monotonically with depth in 10. With the READINGS definition conv < BiLSTM in 2.

### T8. Not fetched or not loaded

- Load errors under kraken 7.0.2: 0.

### T9. What looks related, and what is related

**a) Parent → child (documented and tested, or inferred): weights close** (16 pairs)

| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |
|---|---|---|---:|---:|---:|
| `bern-hertwig-v1` – `cursive_inzigkofen` | — | yes | 0.98 | 1.00 | 0.13 |
| `catmus_print_large` – `french_vietnamese` | — | yes | 0.92 | 1.00 | 0.19 |
| `bern-hertwig-v2` – `bern-schlussredaktion-v1` | Bern (in-house) | yes | 0.99 | 1.00 | 0.20 |
| `bern-hertwig-v1` – `bern-hertwig-v2` | Bern (in-house) | yes | 1.00 | 1.00 | 0.23 |
| `bern-combined-v1` – `bern-hertwig-v2` | Bern (in-house) | yes | 0.99 | 1.00 | 0.23 |
| `mccatmus` – `prima` | — | yes | 0.79 | 1.00 | 0.26 |
| `german-xix-blb-v1` – `german-xix-v1` | German XIX (in-house), German script/print | yes | 0.76 | 1.00 | 0.27 |
| `bern-hertwig-v2` – `cursive_inzigkofen` | — | yes | 0.98 | 1.00 | 0.28 |
| `catmus_print_large` – `latin_incunabula` | — | yes | 0.38 | 0.02 | 0.33 |
| `textualis_inzigkofen` – `tridis_v2` | — | yes | 0.98 | 1.00 | 0.33 |
| `openiti_arabic_print` – `openiti_persian_print` | — | yes | 1.00 | 1.00 | 0.70 |
| `german-xix-v1` – `bohemian_19th` | — | yes | 0.60 | 1.00 | 0.70 |
| `openiti_arabic_print` – `openiti_ottoman_print` | — | yes | 1.00 | 1.00 | 0.72 |
| `bastarda_inzigkofen` – `catmus_medieval` | — | yes | 0.98 | 1.00 | 0.72 |

(first 14 of 16, sorted by distance)

**b) The record names the base and the codec is largely the base's, but the weights are unrelated** (1 pair)

| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |
|---|---|---|---:|---:|---:|
| `bifrost_old_norse` – `catmus_medieval` | — | yes | 0.78 | 0.38 | 1.36 |

**c) Same project or name, same architecture, weights unrelated (distance ≥ 1.2)** (16 pairs)

| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |
|---|---|---|---:|---:|---:|
| `cyrillic_uncial` – `glagolitic_print` | Slavic/Cyrillic | yes | 0.08 | 0.03 | 1.28 |
| `catmus_medieval` – `mccatmus` | CATMuS | yes | 0.38 | 0.01 | 1.35 |
| `latin_expanded_11c` – `latin_graphematic_11c` | Latin 11th c. (Worms) | yes | 0.61 | 0.01 | 1.37 |
| `bifrost_old_norse` – `old_norse_am305` | Old Norse | yes | 0.28 | 0.01 | 1.38 |
| `austrian_fraktur` – `german_print` | German script/print | yes | 0.66 | 0.02 | 1.39 |
| `catmus_gothic_print` – `catmus_print_large` | CATMuS | yes | 0.57 | 0.01 | 1.40 |
| `german-xix-blb-v1` – `mendelssohn_letters` | German script/print | yes | 0.52 | 0.27 | 1.40 |
| `german-xix-v1` – `mendelssohn_letters` | German script/print | yes | 0.52 | 0.29 | 1.41 |
| `cremma_medieval` – `medieval_latin_french_abbreviated` | Medieval Latin/French | yes | 0.45 | 0.01 | 1.41 |
| `fondue_gd_v2` – `manu_mcfondue` | FoNDUE | yes | 0.44 | 0.21 | 1.42 |
| `german-xix-v1` – `german_print` | German script/print | yes | 0.49 | 0.02 | 1.42 |
| `german-xix-blb-v1` – `german_print` | German script/print | yes | 0.47 | 0.02 | 1.42 |
| `german_print` – `mendelssohn_letters` | German script/print | yes | 0.72 | 0.01 | 1.44 |
| `german-xix-v1` – `austrian_fraktur` | German script/print | yes | 0.45 | 0.02 | 1.45 |

(first 14 of 16, sorted by distance)

**d) Weights related, codec rebuilt (overlap < 0.5)** (7 pairs)

| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |
|---|---|---|---:|---:|---:|
| `catmus_print_large` – `latin_incunabula` | — | yes | 0.38 | 0.02 | 0.33 |
| `french_vietnamese` – `latin_incunabula` | — | yes | 0.36 | 0.02 | 0.37 |
| `catmus_medieval` – `old_norse_am305` | — | yes | 0.23 | 0.01 | 0.61 |
| `bastarda_inzigkofen` – `old_norse_am305` | — | yes | 0.23 | 0.01 | 0.83 |
| `glagolitic_print` – `hebrew_sephardi` | — | yes | 0.45 | 1.00 | 0.83 |
| `cursive_inzigkofen` – `old_norse_am305` | — | yes | 0.23 | 0.01 | 0.84 |
| `hebrew_sephardi` – `samaritan` | — | yes | 0.45 | 0.03 | 0.95 |

**e) Siblings of one parent are further from each other than from the parent** (1 pair)

| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |
|---|---|---|---:|---:|---:|
| `bastarda_inzigkofen` – `cursive_inzigkofen` | Inzigkofen | yes | 0.98 | 0.98 | 0.86 |

**f) Same name family, different architecture (weights cannot be shared)** (11 pairs)

| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |
|---|---|---|---:|---:|---:|
| `bastarda_inzigkofen` – `textualis_inzigkofen` | Inzigkofen | no | 0.33 | 0.00 | — |
| `catmus_gothic_print` – `catmus_medieval` | CATMuS | no | 0.77 | 0.00 | — |
| `catmus_gothic_print` – `mccatmus` | CATMuS | no | 0.42 | 0.01 | — |
| `catmus_medieval` – `catmus_print_large` | CATMuS | no | 0.51 | 0.01 | — |
| `catmus_print_large` – `mccatmus` | CATMuS | no | 0.44 | 0.27 | — |
| `cremma_medieval` – `medieval_latin_french_expanded` | Medieval Latin/French | no | 0.40 | 0.01 | — |
| `cursive_inzigkofen` – `textualis_inzigkofen` | Inzigkofen | no | 0.34 | 0.00 | — |
| `estournelles_typewritten` – `french_vietnamese` | French | no | 0.32 | 0.04 | — |
| `estournelles_typewritten` – `peraire_french` | French | no | 0.74 | 0.03 | — |
| `french_vietnamese` – `peraire_french` | French | no | 0.32 | 0.05 | — |
| `medieval_latin_french_abbreviated` – `medieval_latin_french_expanded` | Medieval Latin/French | no | 0.87 | 0.05 | — |

