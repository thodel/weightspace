# Weight survey of the kraken model zoo (read-only)

Written 07.11.2026 on asteraix, completed 13.11.2026 (all 37 models). For the manuscript (`paper/manuscript.md`, Section 6.4). Brief: `docs/BRIEF_ZOO_SURVEY.md`.
Code: `wsp/zoo.py` (survey, lineage), `scripts/zoo_fetch.py` (download), `scripts/zoo_report.py` (tables), `scripts/zoo_rare_rows.py` (measured frequencies), `wsp/viz.py zoo` (figures).
Data: `docs/zoo/survey.json` (everything below), `docs/zoo/lineage.json` (every pair: codec overlap, name family, architecture, distance), `docs/zoo/rare_rows.json`,
`docs/zoo/fetch.json` (DOI and sha256 of every file downloaded), `docs/zoo/gateway_zenodo.json` (the gateway's Zenodo-sourced entries), `docs/zoo/tables.md` (the tables of this file).
No forward pass, no image, no held-out data; kraken 7.0.2 on CPU; model files stay in `models/zoo/` (git-ignored).

## 0. What was surveyed

- **All 37 third-party kraken models of the gateway were downloaded and read, and all loaded under kraken 7.0.2 without error.** The brief and `docs/MODELS.md` say 42; the gateway
  (`gateway_models`, 63 entries) lists 38 entries with a Zenodo DOI, 37 of them kraken models and one the `party` base model. The YAML itself is not on this machine, so the count is the gateway's.
- **The download was in two runs.** On 07.11.2026 the first 30 came through and Zenodo then refused the 31st, `kraken-gallicorpora` (10.5281/zenodo.7410529), with
  `HTTP 429 TOO MANY REQUESTS` (`zenodo.org/oai2d … GetRecord`); following the brief I stopped there and made no retry. On 13.11.2026 I ran the same script again; the seven missing records
  (`gallicorpora`, `openiti_arabic_print`, `openiti_persian_print`, `openiti_ottoman_print`, `lectaurep_french`, `hebrew_sephardi`, `english_print`) downloaded without any refusal. Nothing needed credentials.
- Alongside the 37, 10 in-house files from `models/kraken/` were read: the six models of the article's Table 1 (four Bern, two German XIX) and four comparators (`medieval-shard00-std` with the standard stack,
  `plus-shard00-cr1`, `medieval-shard00-v1`, `collapse-probe-131-cr1`; the `Cr255` probe is left out as in the readings). The seventh model already read, the parent `cursive_inzigkofen`, is the zoo file;
  the in-house copy is byte-identical (sha256 equal). 47 models in all.
- The survey is one snapshot of what the gateway serves, not of the kraken repository, and says nothing about the repository's 48 models in 18 writing systems.

## 1. Method

1. Each model is loaded with `KrakenSubject` (CoreML `.mlmodel`; every one of the 47 is). Reported: VGSL string with block names, dropout rates and output width removed (the *architecture key*),
   tensor and parameter counts, codec size, and the codec split by Unicode category of its first code point into Latin letters, digits, punctuation/symbols, combining marks and other. The metadata keys of the files
   (`accuracy, metrics, seg_type, one_channel_mode, model_type, codec, hyper_params, vgsl`, sometimes `legacy_polygons`, `completed_epochs`) name no base and no training data, so
   **a documented base exists only in the gateway field `base_model`, in README text, in the trainer's job record (for the in-house models), or in `docs/MODELS.md`**; I say which.
2. Output layer (`O_*`): row norms, the blank row (row 0): norm, bias, and the bias **relative to the mean of the other biases** (a constant added to all biases leaves the softmax unchanged, so only the relative value can matter; I report both).
   The ten most frequent letters are a fixed proxy, `e n i a r t s o u d`, tested only on models whose codec contains all ten.
3. Models with the same architecture key form a group. In a group the distance of two models is ‖a − b‖ / mean(‖a‖, ‖b‖) over all non-output tensors (symmetric); independent random networks sit at √2 ≈ 1.41.
   For a parent → child pair the per-tensor relative L2 and cosine are computed as in `wsp.stats.lineage_diff`, with output rows **aligned by character** (`wsp.align.codec_row_map`), never by index.
4. Documented pairs are those named by the gateway `base_model`, a README sentence, the trainer's job record or `docs/MODELS.md` (T3). A pair is tested only if the architectures agree and the weights are related (distance < 1.0).
5. The three observations of `docs/READINGS.md` are turned into counted criteria (T5–T7). For the zoo pairs "rare character" is a proxy (outside basic Latin letters, digits, space and ASCII punctuation; about two thirds of every codec).
   For the in-house pairs, whose training-side text exists, Section 4b repeats observation 1 with measured frequencies.
6. "Looks related" (Section 3b) is measured as a shared name or project (my hand-written families in `wsp.zoo.NAME_FAMILIES`, from the model names and Zenodo titles), the same architecture, and the overlap of the codecs.

## 2. Per-model table, architecture groups

![Blank bias of all 47 models](zoo/figures/zoo-blank-bias.svg#only-light)
![Blank bias of all 47 models](zoo/figures/zoo-blank-bias-dark.svg#only-dark)

*Figure 1. Blank-row bias of every model, raw (left) and relative to the mean of the other biases (right).*

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


**What the two tables say.**

- **The standard stack is the zoo's majority.** 27 of the 37 zoo models (73 %) have exactly the stack of the in-house models (`Cr3,13,32 … Lbx200 … O1c<N>`, 4.0–4.1 M parameters, group A; the in-house models add 7 more: the six of Table 1 and `medieval-shard00-std`).
  Their dropout is written in two ways in the VGSL strings (`Do0.1,2` in 16 files, `Do.{…}1,2` in 11), which is no difference in weights. Ten zoo models differ: six use the group-normalised stack of CATMuS-Print (group B, 5.7 M parameters), one the same with a 48-pixel input
  (`estournelles_typewritten`, group F), two the 140-pixel `Cr4,16,32 … Lbx256` stack of TRIDIS (group C, 6.3 M) and one, `english_print`, is a one-off (group E: 48-pixel input, two 3×3 convolutions, one BiLSTM of 100, 0.73 M parameters). So 36 of 37 fall into three
  families (standard 27, group-normalised 7, TRIDIS 2); `kuzushiji` uses the standard stack with a 16,906-class output layer (10.8 M parameters), the only large codec.
- **Codec composition varies more than architecture.** Latin letters are 0–0.73 of the codec (median 0.38 among zoo models); four codecs have almost none (`kuzushiji`, `samaritan`, `hebrew_sephardi` 0.00, `cyrillic_uncial` 0.08); combining marks 0 to 61.
  The CATMuS-lineage models carry 40–44 combining marks.
- **The blank row is a distinct row in every model.** Its norm is below the mean of the other rows' norms in 47 of 47 (T1, last column), and its bias is the highest in 28 of 47 (21 of 37 in the zoo).

![Distance between the 34 models of the standard stack](zoo/figures/zoo-distance-standard.svg#only-light)
![Distance between the 34 models of the standard stack](zoo/figures/zoo-distance-standard-dark.svg#only-dark)

*Figure 2. Distance (non-output weights) between all 34 models of group A, in average-linkage order. Grey = unrelated (≈ √2); only cells < 1.0 are printed.*

## 3. Lineage: weights against the record

Figure 2 is the main result of this section: **in a group of 34, only a few clusters of related weights exist, and almost everything else is as far apart as random networks** (median pair distance 1.42). Of the 561 pairs of group A, 37 are below 1.0:
the Bern family with `cursive_inzigkofen` and, further out, `catmus_medieval`, `bastarda_inzigkofen` and `old_norse_am305` (0.13–0.89); the three OpenITI models (0.28–0.72); `mccatmus` – `prima` (0.26); `bohemian_19th` with the two German XIX models (0.70–0.74); and `hebrew_sephardi` with `glagolitic_print` (0.83).

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


**Where the weights agree with the record.** `bastarda_inzigkofen` and `cursive_inzigkofen` (CATMuS Medieval, distance 0.72 and 0.74), `textualis_inzigkofen` (TRIDIS v2, 0.33), `french_vietnamese` and `latin_incunabula` (CATMuS-Print Large, 0.19 and 0.33),
`openiti_persian_print` and `openiti_ottoman_print` (OpenITI Arabic, 0.70 and 0.72), `german-xix-v1` (`bohemian_19th`, 0.70) and the six in-house pairs (0.13–0.28): **14 of the 16 documented pairs of T3 are related**; the other two are the contradiction and the architecture mismatch below.
The size of the move is large for the public fine-tunes: the Inzigkofen models are 0.9 (pooled relative L2 on conv and BiLSTM) from CATMuS Medieval, the OpenITI children 0.76–0.86 from their base, against 0.1–0.3 in the in-house pairs; these are heavy retrainings that started from the parent rather than gentle adaptations.

**`german-xix-v1`, a case where the weights came first.** `docs/MODELS.md` listed its base as "to confirm". The first run of this survey found `bohemian_19th` at 0.70 (its 135-character codec contained in `german-xix-v1`'s 224-character codec at identical row indices) and I called this a candidate.
The trainer's job record settles it: `jobs/20261007T131819Z-kraken-german-xix-v1/job.json` has `request.base_model = kraken-bohemian_19th`, `resize: union`, `lrate 1e-4`, cosine schedule, augmentation on, 97 epochs run. (The same request carries the kraken+ `spec`; it cannot have been applied, since the model has the standard stack, so the base's architecture was kept.)
It is documented in T3 now. What the record does not explain is the first convolutional block: relative L2 1.29 between `bohemian_19th` and `german-xix-v1` (the other conv blocks 0.59–0.74), where the other pairs have 0.05–0.26 (0.66 for the OpenITI children). After 97 epochs at lrate 1e-4 the first block is nearly unrelated to the base's while the deeper ones are not; this remains unexplained.

**Where they do not, and one caution.**

1. **`bifrost_old_norse`** (gateway `base_model` CATMuS Medieval) is at 1.36 from `catmus_medieval`, as far as from unrelated models, and has no near neighbour at all. Its Zenodo text names no base. The weights do not support the documented base *as the file now in the zoo*; they do not say what it was trained from. One reading: another version of CATMuS Medieval; I cannot confirm or exclude that from metadata.
2. **`catmus_gothic_print`** (gateway `base_model` CATMuS Medieval; README: "fine-tuned on CATMuS Medieval") has a different architecture (group B, `Gn` stack, 5.7 M parameters, against the 4.1 M standard stack of `catmus_medieval`); a weight-level fine-tune of this file is impossible, and the model is 1.40 from `catmus_print_large`, the other CATMuS print model, in the same group. The record can only refer to another file or to data, not to initial weights.
3. **`cursive_inzigkofen`** (caution, not a contradiction) is documented as a fine-tune of CATMuS Medieval but ranks its parent only 5th of 33 in distance (0.74): the four Bern models are closer to it (0.13–0.37), as they should be, and `old_norse_am305` (0.84) is about as close as the parent.

**Relations that no record names (T4).**

- **`mccatmus` → `prima`:** distance 0.26; `prima`'s 146-character codec contains all 116 characters of `mccatmus` at identical row indices. Weights and codec say `prima` extends `mccatmus` (or both extend a third model that is not in the set); the Zenodo record of `prima` has no text file.
- **`hebrew_sephardi` → `glagolitic_print`:** distance 0.83; the 70-character codec of `hebrew_sephardi` (Hebrew letters and points) is contained in the 154-character codec of `glagolitic_print` at identical row indices, which no script reason explains. Direction from the codec, nothing in the Zenodo text of either; the weights of the Glagolitic model are 0.72 conv / 1.16 BiLSTM (pooled) from the Hebrew one, so the recurrent stack was largely retrained.
  `samaritan` is a weak third member (0.95 to `hebrew_sephardi`, 1.02 to `glagolitic_print`); I make nothing of it.
- **`old_norse_am305` ↔ `catmus_medieval`** (0.61, the closest pair of the standard stack after the in-house ones): the README says "fine-tuned from a base model trained on AM 302 fol", a model not in the gateway list; its codec shares only 99 characters with CATMuS Medieval and none at the same row index (1 of 99). The base of the base is presumably in the CATMuS line; the weights agree, the codec was rebuilt.
- **`openiti_persian_print` – `openiti_ottoman_print`:** 0.28, closer to each other than either is to their documented parent (0.70, 0.72). Both are documented fine-tunes of `openiti_arabic_print`; the shortcut would be a shared intermediate step, but nothing in the text says so.
- **No re-upload and no near-duplicate:** no two of the 47 files share a sha256 (apart from the parent copy, which is the same file), and no two have a distance below 0.02.

**Documented but not testable here:** `kuzushiji` (finetuned on top of the Chat model by Colin Brisson), `mendelssohn_letters` (a German manuscripts base model from eScriptorium, not identified by a DOI), `old_norse_am305` (a model on AM 302 fol), `peraire_french` (Manu McFrench, Zenodo 6657809). None has its parent in the set;
`kuzushiji`, `mendelssohn_letters` and `peraire_french` sit at 1.36–1.42 from everything in their group, as an untested parent would leave them.

## 3b. Lineage drawn: what looks related against what is related

![Lineage of the surveyed models](zoo/figures/zoo-lineage.svg#only-light)
![Lineage of the surveyed models](zoo/figures/zoo-lineage-dark.svg#only-dark)

*Figure 3a. Every parent → child relation of this section. Green: the record names the parent and the weights agree (distance printed; 1.41 = unrelated). Orange dashed: the record names a parent and the weights do not support it (`bifrost_old_norse`) or the architecture rules it out (`catmus_gothic_print`).
Blue dotted: only weights and codec suggest it (arrows: direction from the codec). Grey, dashed: documented parents that are not in the surveyed set. Orange boxes: in-house models.*

![Appearance against weights](zoo/figures/zoo-appearance.svg#only-light)
![Appearance against weights](zoo/figures/zoo-appearance-dark.svg#only-dark)

*Figure 3b. Left: one dot per pair of models with the same architecture (578 pairs); x = overlap of the codecs, y = weight distance. Right: the 28 pairs outside the in-house models that share a project or a name (families in `wsp.zoo.NAME_FAMILIES`).*

"Looks related" is measured three ways: a shared name or project, the same architecture, and the overlap of the codecs (Jaccard of the character sets; and the share of shared characters that sit at the same output row, which is 1.00 when a child's codec extends its parent's by appending). "Is related" is the weight distance of Section 3.

**What the comparison shows**

1. **A shared name or project is a poor guide.** Of the 17 pairs with a shared name family, the same architecture and no in-house model on both sides, **1 is related by weights** (`bastarda_inzigkofen` – `cursive_inzigkofen`, 0.86); 16 sit at 1.28–1.49, i.e. unrelated
   (T9 c): the two Latin 11th-century models of one project (1.37), `german_print` – `mendelssohn_letters` (1.44), `mccatmus` – `catmus_medieval` (1.35), `catmus_gothic_print` – `catmus_print_large` (1.40), `fondue_gd_v2` – `manu_mcfondue` (1.42). Eleven more pairs
   in a name family cannot share weights at all because their architectures differ (T9 f), among them the three Inzigkofen models, whose `textualis_inzigkofen` descends from TRIDIS, not from CATMuS Medieval like its namesakes, and `medieval_latin_french_abbreviated` – `…_expanded` (same project, codec overlap 0.87, different architectures).
2. **A near-identical codec is a reliable sign of shared weights; a partial one is not.** Of the 26 same-architecture pairs whose codecs overlap by 0.9 or more, all 26 are related by weights. Of the 508 pairs below 0.5, 12 are related, so a low overlap does not exclude descent
   (`latin_incunabula` kept 121 of its parent's 220 characters, added 102 and re-ordered them, yet its weights are 0.33 from `catmus_print_large`; `old_norse_am305` shares 23 % of its codec with CATMuS Medieval and is 0.61 from it; `glagolitic_print` overlaps 0.45 with `hebrew_sephardi` only because it is the larger codec; T9 d).
   In between (0.5–0.9, 44 pairs) only 3 are related.
3. **A child's codec usually extends its parent's in place.** In 15 of the 16 parent → child pairs of T9 a every character the parent has sits at the same output row in the child (index agreement 1.00); the exception is `latin_incunabula` (0.02). The agreement is evidence for descent and is how `mccatmus` → `prima`, `hebrew_sephardi` → `glagolitic_print`
   and `bohemian_19th` → `german-xix-v1` were found, but its absence proves nothing. It is informative only when many characters are shared.
4. **A documented base with a reused character set and unrelated weights:** `bifrost_old_norse` overlaps 0.78 with the codec of `catmus_medieval` (a true child has 0.96–0.98) but only 38 % of the shared characters sit at the same row and the weights are 1.36 away. The character inventory was taken over, the weights were not, or the model started from another version.
   The metadata does not say which; the claim "fine-tuned from CATMuS Medieval" cannot be confirmed from this file.
5. **Siblings are no safe guide either way.** `bastarda_inzigkofen` and `cursive_inzigkofen` are both fine-tunes of CATMuS Medieval with codecs that overlap by 0.98; by codec they look like the closest pair of the group. By weights they are 0.86 from each other and 0.72 and 0.74 from their common parent: two heavy fine-tunes of one parent are further apart than either is from the parent.
   The OpenITI children show the opposite: `persian` and `ottoman` are 0.28 apart and 0.70 and 0.72 from `openiti_arabic_print`. Sibling distance depends on how the fine-tunes were made, and the zoo does not record that.
6. **Where the lineage comes from.** Of the 16 documented pairs of T3, 14 are confirmed by weights, 1 contradicted (`bifrost_old_norse`), 1 excluded by architecture (`catmus_gothic_print`); two further relations are weights-and-codec only; four models name a parent that is outside the set; 14 of the 37 zoo models have no relation to any other surveyed model
   (the list under Figure 3a), and three more (`kuzushiji`, `mendelssohn_letters`, `peraire_french`) have a documented parent that is not in the set and no related model in it. That the weights agree with so many records is the result; that names, projects and partial codec overlaps do not is the caution for anyone who would reconstruct lineage from the zoo's titles.

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


## 4. The three observations of `docs/READINGS.md` on the zoo's pairs

Important about the test: the observations were *formulated on six in-house pairs*, so those pairs cannot test them. The **independent test set is the seven documented zoo pairs** (`catmus_medieval` → `bastarda_inzigkofen`, `cursive_inzigkofen`; `tridis_v2` → `textualis_inzigkofen`;
`catmus_print_large` → `french_vietnamese`, `latin_incunabula`; `openiti_arabic_print` → `openiti_persian_print`, `openiti_ottoman_print`), four different parents, three architectures; `bohemian_19th` → `german-xix-v1` is a further pair that was not used to formulate them. Counts below give both: the seven zoo pairs and all 14 documented related pairs
(in-house included). The two weights-only pairs are listed but not counted.

![Parent-child pairs: which stack moved, and the blank bias](zoo/figures/zoo-pairs.svg#only-light)
![Parent-child pairs: which stack moved, and the blank bias](zoo/figures/zoo-pairs-dark.svg#only-dark)

*Figure 4. Pooled relative L2 by stack (left) and the change of the blank bias, raw and relative to the other biases (right).*

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


### Reading of the observations

**(2) The blank bias — holds, and is stronger than the raw value shows.** Relative to the mean of the other biases the blank bias rises in 14 of 14 documented pairs (7 of 7 zoo pairs); the raw bias rises in 13 of 14 (6 of 7; `french_vietnamese` −0.009).
The relative value is mostly larger because *the other biases fall*: their mean dropped in 13 of 14 pairs, by 0.05–0.84 (0.10–0.60 in the zoo pairs), so in the in-house readings "blank bias +0.06…+0.12" is, for the softmax, a rise of +0.16…+0.57. The exception is `openiti_persian_print`, where the other biases rose by 0.05 and the relative rise (+0.21) is smaller than the raw (+0.26).
For the editing sessions this matters: setting `bias[0]` back by its raw Δ undoes only part of the change in the quantity that decides the outputs. Across the 47 models without pairs: relative blank bias > 0 in 45; **the two negative ones are the two models with a 48-pixel input**, `estournelles_typewritten` (−1.10, blank last of 106 by bias) and `english_print` (−0.67); the blank has the highest bias in 28.
The in-house relative values (T1): `hertwig-v1` +1.80, `hertwig-v2` +2.00, `schlussredaktion` +2.39, `combined` +2.58, `xix-v1` +2.99, `xix-blb` +3.15, `shard00-std` +0.33 and the parent `cursive_inzigkofen` +1.48 (the zoo median is 0.98, the in-house median 1.90). Whether a pair's rise *causes* deletions is not something weights show.

**(1) Rare rows — holds in direction, not in selectivity, and fails for one pair.** With the proxy, among the special rows the norm grew in 13 of 14 pairs (6 of 7 zoo pairs; the exception `latin_incunabula` has only 51 shared special rows) and the bias fell in a majority in 14 of 14; the median bias change of special rows is negative in every pair (−0.06 to −2.02; −0.06 to −0.84 in the zoo pairs), that of the frequent letters ranges −0.31 to +0.14.
But "the rows that moved most are the rare ones" holds in only 7 of 14 by median movement (3 of 7 zoo pairs) and in 9 of 14 by the top decile (2 of 7), and all four criteria hold in 5 of 14 (**zoo 1 of 7**). In `tridis_v2` → `textualis_inzigkofen`, `catmus_print_large` → `french_vietnamese` and both OpenITI pairs the frequent letters moved *more* than the special rows.
The median norm ratio of special rows is 0.85–1.56 over the 14 pairs (1.41 and 1.52 for the two Inzigkofen models); the 30–110 % growth of the first readings concerned the rows that moved most and is not tabulated here. The companion claim, that frequent letters have the weakest rows, holds without any pair: the mean norm of the ten frequent rows is below that of all others in 40 of 43 models that contain the ten letters (median ratio 0.56);
the three exceptions are the OpenITI models (printed Arabic-script models, where Latin letters are the rare ones), and at least five of the ten weakest non-blank rows are frequent letters in 29 of 43. The proxy is crude; Section 4b repeats the test with measured frequencies where text exists.

**(3) Conv moves less than the recurrent stack and the output layer — not supported as stated.** Pooled over weights, the conv stack moved less than the BiLSTMs in 11 of 14 pairs (zoo 4 of 7) and less than the output rows in 4 of 14 (zoo 2 of 7); in the heavy `catmus_medieval` retrainings the three are about equal (0.90, 0.88–0.90, 0.70–0.81),
in `textualis_inzigkofen` the conv is highest of the three, and in `glagolitic_print` (inferred) the BiLSTMs moved most. **What does hold is a depth pattern**: the first conv block moves least of the four in 12 of 14 pairs (6 of 7 zoo pairs; relative L2 0.05–0.26, and 0.66 for the two OpenITI children), the exceptions being `openiti_persian_print` and `bohemian_19th` → `german-xix-v1` (1.29), and the conv blocks increase with depth in 10 of 14.
READINGS states the contrast with per-tensor means that include conv biases (tiny norms, large ratios) and the zero `bias_ih` of the LSTMs (counted as no movement); under that definition conv < BiLSTM in only 2 of 14 pairs. Note that the seed null (`docs/SEED_NULL.md`) shows the same depth pattern between two runs that differ only by seed, so it is a property of how these networks move under any retraining, not a trace of fine-tuning.

### 4b. Observation 1 with measured frequencies (in-house pairs)

For the six in-house pairs the child's training-side text exists (`data/practice/*.jsonl`, 300 lines per set, seed 20261106), so rarity can be counted instead of proxied: each shared character is *absent* (0 occurrences in the 300 lines), *rare* (1–9) or *frequent* (≥ 10). For the zoo pairs no text is available. Counts come from 300 lines, not from the whole training set
(the Hertwig set has 1,786 lines), so some "absent" characters do occur in training; that blurs the contrast and cannot create it.

| pair | practice set | class | rows | median rel. move | median norm ratio | median bias Δ | norm up | bias down |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `cursive_inzigkofen` → `bern-hertwig-v1` | bern-hertwig-train-300 | absent (0) | 185 | 0.28 | 1.16 | -0.41 | 184 | 184 |
| `cursive_inzigkofen` → `bern-hertwig-v1` | bern-hertwig-train-300 | rare (1–9) | 33 | 0.11 | 1.00 | -0.02 | 18 | 17 |
| `cursive_inzigkofen` → `bern-hertwig-v1` | bern-hertwig-train-300 | frequent (≥10) | 43 | 0.14 | 0.98 | -0.00 | 13 | 22 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | bern-hertwig-train-300 | absent (0) | 185 | 0.41 | 1.23 | -0.72 | 184 | 179 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | bern-hertwig-train-300 | rare (1–9) | 33 | 0.24 | 1.04 | +0.02 | 24 | 15 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | bern-hertwig-train-300 | frequent (≥10) | 43 | 0.25 | 1.01 | -0.00 | 24 | 23 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | bern-hertwig-train-300 | absent (0) | 187 | 0.14 | 1.07 | -0.32 | 186 | 179 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | bern-hertwig-train-300 | rare (1–9) | 35 | 0.17 | 1.03 | +0.04 | 29 | 12 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | bern-hertwig-train-300 | frequent (≥10) | 44 | 0.17 | 1.03 | -0.00 | 34 | 24 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | bern-train-300 | absent (0) | 193 | 0.12 | 1.07 | -0.53 | 178 | 177 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | bern-train-300 | rare (1–9) | 31 | 0.15 | 1.02 | -0.01 | 21 | 21 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | bern-train-300 | frequent (≥10) | 42 | 0.16 | 0.98 | -0.02 | 12 | 29 |
| `bern-hertwig-v2` → `bern-combined-v1` | bern-train-300 | absent (0) | 193 | 0.14 | 1.08 | -0.74 | 176 | 173 |
| `bern-hertwig-v2` → `bern-combined-v1` | bern-train-300 | rare (1–9) | 31 | 0.17 | 1.02 | -0.03 | 19 | 20 |
| `bern-hertwig-v2` → `bern-combined-v1` | bern-train-300 | frequent (≥10) | 42 | 0.20 | 0.97 | -0.02 | 12 | 28 |
| `german-xix-v1` → `german-xix-blb-v1` | blb-train-300 | absent (0) | 134 | 0.17 | 1.02 | -0.75 | 102 | 100 |
| `german-xix-v1` → `german-xix-blb-v1` | blb-train-300 | rare (1–9) | 20 | 0.25 | 1.00 | +0.05 | 10 | 8 |
| `german-xix-v1` → `german-xix-blb-v1` | blb-train-300 | frequent (≥10) | 70 | 0.17 | 1.01 | +0.05 | 42 | 14 |

| pair | ρ(log count, rel. move) | ρ(log count, norm ratio) | ρ(log count, bias Δ) |
|---|---:|---:|---:|
| `cursive_inzigkofen` → `bern-hertwig-v1` | -0.64 (p=0.000) | -0.69 (p=0.000) | +0.75 (p=0.000) |
| `cursive_inzigkofen` → `bern-hertwig-v2` | -0.59 (p=0.000) | -0.67 (p=0.000) | +0.73 (p=0.000) |
| `bern-hertwig-v1` → `bern-hertwig-v2` | +0.35 (p=0.000) | -0.41 (p=0.000) | +0.70 (p=0.000) |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | +0.36 (p=0.000) | -0.48 (p=0.000) | +0.57 (p=0.000) |
| `bern-hertwig-v2` → `bern-combined-v1` | +0.36 (p=0.000) | -0.56 (p=0.000) | +0.56 (p=0.000) |
| `german-xix-v1` → `german-xix-blb-v1` | +0.02 (p=0.823) | -0.17 (p=0.013) | +0.44 (p=0.000) |


**Reading.** The effect is not graded by frequency; it is a step at *absent*. In all six pairs the rows of characters that never occur in the child's text (134–193 rows) have a median bias change of −0.32 to −0.75 (−0.41, −0.72, −0.32, −0.53, −0.74, −0.75) and a norm that grows in almost every row (184 of 185 for the first Bern adaptation; 102 of 134 for the Kurrent pair).
Characters that occur 1–9 times and characters that occur ten times or more show a median bias change of −0.03 to +0.05 and a median norm ratio of 0.97–1.04: they move as the rest of the network moves. The Spearman correlation between log count and the bias change is +0.44 to +0.75 in every pair (p < 0.001), and between log count and the norm ratio −0.17 to −0.69.
What does **not** hold uniformly is that these rows move *more*: in the first adaptation from the foreign parent (`cursive_inzigkofen` → `hertwig-v1`, `hertwig-v2`) the correlation of log count with the relative move is −0.64 and −0.59, but in the three continuations (`v1` → `v2`, → `schlussredaktion`, → `combined`) it is +0.35 to +0.36 and in the Kurrent pair +0.02: rows of absent characters that were already pushed down by an earlier fine-tune
move less afterwards than the characters that are in use. So the sentence of the first readings should read: *rows of characters that do not occur in the child's text lose bias and, in the first adaptation, gain norm; a few occurrences are enough to keep a row with the rest of the network.* That is what CTC training does to a class that only receives negative gradient, now with counts to show it.

## 5. Outliers

| model | what is unusual | numbers |
|---|---|---|
| `kuzushiji` | 16,906 output classes, 10.8 M parameters, only non-Latin codec at that size | blank rel +5.92, rank 1 |
| `english_print` | 0.73 M parameters, one BiLSTM of 100, 48-pixel input; a model of its own; blank last-ish by bias | raw −0.69, rel −0.67, rank 83 of 90 |
| `estournelles_typewritten` | 48-pixel input; the most negative blank bias and a blank that ranks last | raw −1.04, rel −1.10, rank 106 of 106; mean row norm 33.0 |
| `mendelssohn_letters`, `textualis_inzigkofen`, `tridis_v2` | output rows with norms 53–60 against a zoo median of 10.5 | blank rel +8.9, +7.5, +6.8; blank rank 1, 32, 61 |
| `bifrost_old_norse` | documented base not supported by weights | distance to documented parent 1.36 |
| `catmus_gothic_print` | documented base has another architecture | group B vs group A |
| `german-xix-v1` ← `bohemian_19th` | documented base (trainer record) and related weights, first conv block not | C_0 relative L2 1.29 |
| `glagolitic_print` ← `hebrew_sephardi` | codec contains the Hebrew model's at the same rows; no text says so | distance 0.83; BiLSTM 1.16 |
| `latin_incunabula` | only zoo pair where special rows mostly do not grow (15 of 51); keeps 121 of its parent's 220 characters, adds 102 | output rows 0.63 against conv 0.26 |
| `fondue_gd_v2` | second-highest raw blank bias of all models (after `kuzushiji`, +2.00); 2 completed epochs in its metadata | raw +1.88, rel +2.34 |
| `gallicorpora`, `latin_expanded_11c`, `latin_graphematic_11c`, `plus-shard00-cr1` | blank bias hardly above the other biases | rel +0.13, +0.10, +0.11, +0.09 |

The collapse probe (+0.02) is the collapsed run of the readings (val accuracy 0.0083); I have no information on why the others look alike.

## 6. Open questions

1. Why is the first convolutional block of `german-xix-v1` nearly unrelated to its documented base `bohemian_19th` (1.29) while the deeper blocks are not? The job record gives the recipe; it does not explain this.
2. Which file did `bifrost_old_norse` start from? An earlier CATMuS Medieval release would explain both the 1.36 and the record.
3. `prima` ← `mccatmus` and `glagolitic_print` ← `hebrew_sephardi` rest on weights and codec; the authors of the models, or their Zenodo records beyond what the gateway shows, would confirm or correct them.
4. Rare-row selectivity in the zoo pairs: needs the training lines of a few zoo models (not available here, and no ground truth was to be downloaded).
5. Why are the two OpenITI children 0.28 from each other and 0.70 from their parent, and the two Inzigkofen fine-tunes 0.86 from each other and 0.72 from theirs? Sibling distance seems to depend on how the fine-tunes were made.

## 7. What this means for Section 6.4 of the manuscript

Section 6.4 can say, and the tables above support, the following. **First**, the standard stack is the zoo's majority: 27 of the 37 third-party models (73 %) have exactly the architecture of the article's models and 36 of 37 fall into three families; the survey covers all 37 models the gateway lists (the first run had 30; Zenodo refused the rest at the time).
**Second**, the weights agree with the documented lineage where it can be tested (14 of 16 documented pairs, one clear contradiction, one architecture mismatch); in a group of 34 models only a few clusters are related at all, and with ≈ 1.41 as the null, lineage is recoverable for fine-tunes that did not move far (0.1–0.3) and visible but less clean for heavy retrainings (0.7–0.9).
The weights also found two relations that no record names and one (`german-xix-v1`) that the trainer's record then confirmed. A shared name, project or partial codec overlap does not predict lineage (1 of 17 name-sharing pairs is related); a near-identical codec does (26 of 26). **Third**, the blank-bias observation generalises when stated relative to the other biases (14 of 14 pairs, 7 of 7 independent zoo pairs; raw 13 of 14);
the article's raw "+0.06 … +0.12" understates it. **Fourth**, the other two must be weakened: rare-row inflation holds in direction with the proxy (norm up in 13 of 14, bias down in 14 of 14) but not as "the rows that moved most" (all four criteria in 1 of 7 independent pairs); with measured frequencies it is a step at characters that never occur in the child's text, not a gradient by frequency;
and the claim that the conv stack moves less than the recurrent stack and the output layer is not supported (conv < BiLSTM in 4 of 7 independent pairs, conv < output in 2 of 7), while the first conv block moves least in 12 of 14, which the seed null shows to occur between seeds as well.
What the survey cannot say: that the zoo's models were trained the way the in-house ones were, that "rare" is measured for the zoo pairs (it is a proxy there), that any weight pattern predicts CER (nothing here was scored), or what happened to the models' training data.

## Reproduction

```bash
export CUDA_VISIBLE_DEVICES="" TMPDIR=~/atr-cache/tmp
.venv/bin/python scripts/zoo_fetch.py docs/zoo/gateway_zenodo.json models/zoo docs/zoo/fetch.json   # resumes; stops on 401/403/429
.venv/bin/python -m wsp.zoo survey --inhouse models/kraken --out docs/zoo/survey.json                 # ~2 min on CPU
.venv/bin/python -m wsp.zoo lineage --inhouse models/kraken --out docs/zoo/survey.json                # writes docs/zoo/lineage.json
.venv/bin/python scripts/zoo_report.py docs/zoo/survey.json > docs/zoo/tables.md
.venv/bin/python scripts/zoo_rare_rows.py models/kraken docs/zoo/rare_rows.json > docs/zoo/rare_rows.md
.venv/bin/python -m wsp.viz zoo --out docs/zoo/figures
```
