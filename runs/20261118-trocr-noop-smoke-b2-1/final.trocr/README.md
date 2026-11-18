---
library_name: trocr
pipeline_tag: image-to-text
tags:
- htr
- ocr
- handwritten-text-recognition
- historical-documents
- trocr
base_model: dh-unibe/trocr-kurrent-XVI-XVII
datasets:
- dh-unibe/image-text_zh-regierungsratsprotokolle
- dh-unibe/image-text_kurrent-xix
- dh-unibe/image-text_rats-und-richtebuecher_xv-xvi
- dh-unibe/image-text_koenigsfelden-charters-post-1500
- dh-unibe/image-text_bullinger-autoren
- dh-unibe/image-text_aaeb-xiv-xvii
- dh-unibe/image-text_historisches-grundbuch-basel_xix-xx_train
- dh-unibe/image-text_german-20th-century
- dh-unibe/image-text_parlamentsdienste-protokolle
- dh-unibe/image-text_nr-sr-vereinigte-bundesversammlung-xix
- dh-unibe/image-text_koenigsfelden-charters-part-2
- dh-unibe/image-text_koenigsfelden-adhr-colmar
- dh-unibe/image-text_aaeb-xiv-xvii-part-2
- dh-unibe/image-text_sg-missiven
- dh-unibe/image-text_reichsanzeiger-gt
- dh-unibe/image-text_gt-fraktur
- dh-unibe/image-text_fibeln
- dh-unibe/image-text_dtgt
- dh-unibe/image-text_dach-gt
- dh-unibe/image-text_weisthuemer
metrics:
- cer
- wer
---

# trocr-german-kurrent-xvi-xix-from-kurrent-v2

Handwritten-text-recognition model trained with [`training-atr-models`](https://github.com/thodel/training-atr-models). These are the weights of the **best validation checkpoint** of the run below — not its last epoch.

## Evaluation

| metric | value |
|---|---|
| split CER | 8.89 % |
| split WER | 24.71 % |
| samples scored | 500 |
| characters scored | 23736 |
| character errors | 2109 |

Measured on **this run's own held-out validation split** (page-level and seeded, so no page contributes lines to both sides). It is not a score on a shared benchmark and does not transfer to a different corpus.

**The validation split is a seeded partition of the training projects.** It measures the model on pages it has not seen, in hands it has. Expect a higher error rate on a new writer.

## Training data

- [`dh-unibe/image-text_zh-regierungsratsprotokolle`](https://huggingface.co/datasets/dh-unibe/image-text_zh-regierungsratsprotokolle) @ `f6a06745bb134044d5f937b22000128cc4bc24fd`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_kurrent-xix`](https://huggingface.co/datasets/dh-unibe/image-text_kurrent-xix) @ `20743e9f6982460b243e014091626da43e2b57f7`
  - Training projects: `MM_2_033`, `MM_2_034`, `MM_2_035`, `MM_2_036`, `MM_2_037`, `MM_2_038`, `MM_2_039`, `MM_2_040`, `MM_2_041`, `MM_2_042`, `MM_2_043`, `MM_2_044`, `MM_2_045`, `MM_2_046`, `MM_2_047`, `MM_2_048`, `MM_2_049`, `MM_2_050`, `TEST_CITlab_Bassermann_0_4`, `TEST_CITlab_Bassermann_Manuscripts`, `TEST_CITlab_Bassermann_Manuscripts_0_2`, `TEST_CITlab_Binder_Kochbuch_2`, `TEST_CITlab_Binder_Kochbuch_M3+`, `TEST_CITlab_DreierM2`, `TEST_CITlab_Escher_M1`, `TEST_CITlab_GemeindeChronikM1`, `TEST_CITlab_GrimmBriefe`, `TEST_CITlab_Gusbeth`, `TEST_CITlab_Haeckermann_3+`, `TEST_CITlab_Handschriftliche_Archivquellen`, `TEST_CITlab_Hoffmann,_Hans`, `TEST_CITlab_Konzilsprotkolle_B_Schwartz_5_2018`, `TEST_CITlab_Konzilsprotokolle_A_Haeckermann_2`, `TEST_CITlab_Konzilsprotokolle_A_Haeckermann_3`, `TEST_CITlab_Konzilsprotokolle_M4_HTR+`, `TEST_CITlab_Lotte1`, `TEST_CITlab_Lotte_Tagebuecher`, `TEST_CITlab_Lottes_Schrift`, `TEST_CITlab_MargareteSick`, `TEST_CITlab_MargareteSick20180731`, `TEST_CITlab_MargareteSick20180731a`, `TEST_CITlab_Minutes_of_Estonian_Knighthood`, `TEST_CITlab_MÜLLER`, `TEST_CITlab_MÜLLER_A`, `TEST_CITlab_Nekrolog`, `TEST_CITlab_Nekrolog_2`, `TEST_CITlab_Nekrolog_M2`, `TEST_CITlab_Nekrolog_M3`, `TEST_CITlab_OEAW_M4+`, `TEST_CITlab_Protokoll_Hoftheater_1806`, `TEST_CITlab_Protokoll_Hoftheater_1806-Vers_2018Nov`, `TEST_CITlab_Protokoll_Hoftheater_1806-Vers_2018Nov+`, `TEST_CITlab_Pyl_M1`, `TEST_CITlab_Pyl_M2`, `TEST_CITlab_Pyl_M3`, `TEST_CITlab_Pyl_M4`, `TEST_CITlab_Pyl_M5`, `TEST_CITlab_Rehlen_1834_a`, `TEST_CITlab_Roland-Hilde_M1+`, `TEST_CITlab_Roland_M3+`, `TEST_CITlab_RvE_Barlaam_HS_D_`, `TEST_CITlab_Schwartz_M6`, `TEST_CITlab_St_Aegidi_M1`, `TEST_CITlab_Steiner`, `TEST_CITlab_Suppes`, `TEST_CITlab_Suppes_3500`, `TEST_CITlab_Tagebuch_Arnold_v1`, `TEST_CITlab_Todesurteile_1849-1891`, `TEST_CITlab_Todesurteile_M1+`, `TEST_CITlab_Tuempel_M3`, `TEST_CITlab_umkc_Roland_M1`, `TEST_CITlab_umkc_Roland_M2`, `TRAINING_TESTSET_Binder_Kochbuch_M3+`, `TRAINING_TESTSET_Haeckermann_3+`, `TRAINING_TESTSET_OEAW_M4+`, `TRAINING_TESTSET_Protokoll_Hoftheater_1806-Vers_2018Nov+`, `TRAINING_TESTSET_Roland-Hilde_M1+`, `TRAINING_TESTSET_Roland_M3+`, `TRAINING_TESTSET_Todesurteile_1849-1891`, `TRAINING_TESTSET_Todesurteile_M1+`, `TRAIN_CITlab_Bassermann_0_4`, `TRAIN_CITlab_Bassermann_Manuscripts`, `TRAIN_CITlab_Bassermann_Manuscripts_0_2`, `TRAIN_CITlab_Binder_Kochbuch_2`, `TRAIN_CITlab_Binder_Kochbuch_M3+`, `TRAIN_CITlab_DreierM2`, `TRAIN_CITlab_Escher_M1`, `TRAIN_CITlab_GemeindeChronikM1`, `TRAIN_CITlab_GrimmBriefe`, `TRAIN_CITlab_Gusbeth`, `TRAIN_CITlab_Haeckermann_3+`, `TRAIN_CITlab_Handschriftliche_Archivquellen`, `TRAIN_CITlab_Hoffmann,_Hans`, `TRAIN_CITlab_Konzilsprotkolle_B_Schwartz_5_2018`, `TRAIN_CITlab_Konzilsprotokolle_A_Haeckermann_2`, `TRAIN_CITlab_Konzilsprotokolle_A_Haeckermann_3`, `TRAIN_CITlab_Konzilsprotokolle_M4_HTR+`, `TRAIN_CITlab_Lotte1`, `TRAIN_CITlab_Lotte_Tagebuecher`, `TRAIN_CITlab_Lottes_Schrift`, `TRAIN_CITlab_MargareteSick`, `TRAIN_CITlab_MargareteSick20180731`, `TRAIN_CITlab_MargareteSick20180731a`, `TRAIN_CITlab_Minutes_of_Estonian_Knighthood`, `TRAIN_CITlab_MÜLLER`, `TRAIN_CITlab_MÜLLER_A`, `TRAIN_CITlab_Nekrolog`, `TRAIN_CITlab_Nekrolog_2`, `TRAIN_CITlab_Nekrolog_M2`, `TRAIN_CITlab_Nekrolog_M3`, `TRAIN_CITlab_OEAW_M4+`, `TRAIN_CITlab_Protokoll_Hoftheater_1806`, `TRAIN_CITlab_Protokoll_Hoftheater_1806-Vers_2018Nov`, `TRAIN_CITlab_Protokoll_Hoftheater_1806-Vers_2018Nov+`, `TRAIN_CITlab_Pyl_M1`, `TRAIN_CITlab_Pyl_M2`, `TRAIN_CITlab_Pyl_M3`, `TRAIN_CITlab_Pyl_M4`, `TRAIN_CITlab_Pyl_M5`, `TRAIN_CITlab_Rehlen_1834_a`, `TRAIN_CITlab_Roland-Hilde_M1+`, `TRAIN_CITlab_Roland_M3+`, `TRAIN_CITlab_RvE_Barlaam_HS_D_`, `TRAIN_CITlab_Schwartz_M6`, `TRAIN_CITlab_St_Aegidi_M1`, `TRAIN_CITlab_Steiner`, `TRAIN_CITlab_Suppes`, `TRAIN_CITlab_Suppes_3500`, `TRAIN_CITlab_Tagebuch_Arnold_v1`, `TRAIN_CITlab_Todesurteile_1849-1891`, `TRAIN_CITlab_Todesurteile_M1+`, `TRAIN_CITlab_Tuempel_M3`, `TRAIN_CITlab_umkc_Roland_M1`, `TRAIN_CITlab_umkc_Roland_M2`, `hufeland_privatbesitz_1829`, `nn_msgermqu2124_1827`, `nn_msgermqu2345_1827`, `parthey`, `semper_20_MS`
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_rats-und-richtebuecher_xv-xvi`](https://huggingface.co/datasets/dh-unibe/image-text_rats-und-richtebuecher_xv-xvi) @ `011e82080411a51bcb9e032e9329c919682e5f6c`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_koenigsfelden-charters-post-1500`](https://huggingface.co/datasets/dh-unibe/image-text_koenigsfelden-charters-post-1500) @ `fac683bee395bad466c464bfffd0e6f5299d6c4d`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_bullinger-autoren`](https://huggingface.co/datasets/dh-unibe/image-text_bullinger-autoren) @ `85ea246a5021a5f244c392f97f2c8ac9a42cc717`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_aaeb-xiv-xvii`](https://huggingface.co/datasets/dh-unibe/image-text_aaeb-xiv-xvii) @ `2a9b66116db32c728fd893c3c9af5c41a3fa856d`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_historisches-grundbuch-basel_xix-xx_train`](https://huggingface.co/datasets/dh-unibe/image-text_historisches-grundbuch-basel_xix-xx_train) @ `897509cae41d53fd43d9f5cd0ad5a0e8079e5893`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_german-20th-century`](https://huggingface.co/datasets/dh-unibe/image-text_german-20th-century) @ `47a16a5077a062942237f9817471f60a60993689`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_parlamentsdienste-protokolle`](https://huggingface.co/datasets/dh-unibe/image-text_parlamentsdienste-protokolle) @ `5702670446f335a54505be3d238a5998cf844a04`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_nr-sr-vereinigte-bundesversammlung-xix`](https://huggingface.co/datasets/dh-unibe/image-text_nr-sr-vereinigte-bundesversammlung-xix) @ `de8ad0f86cf878607eb42390ed637f4e1be70b59`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_koenigsfelden-charters-part-2`](https://huggingface.co/datasets/dh-unibe/image-text_koenigsfelden-charters-part-2) @ `1d67552f4470c9d9b11ffb09b0b6a104eef185d1`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_koenigsfelden-adhr-colmar`](https://huggingface.co/datasets/dh-unibe/image-text_koenigsfelden-adhr-colmar) @ `3cffcc9143da1b8248316a9605bda8680887da67`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_aaeb-xiv-xvii-part-2`](https://huggingface.co/datasets/dh-unibe/image-text_aaeb-xiv-xvii-part-2) @ `ce2739fce41c0b8f8673127525839842411020b0`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_sg-missiven`](https://huggingface.co/datasets/dh-unibe/image-text_sg-missiven) @ `495fc94c5f616d9a9161917021c75e7d6f0bf258`
  - Training projects: `sg-missiven`
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_reichsanzeiger-gt`](https://huggingface.co/datasets/dh-unibe/image-text_reichsanzeiger-gt) @ `9ede9f6c5a3c29e762040b2922ef55355caa2ef1`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 10
- [`dh-unibe/image-text_gt-fraktur`](https://huggingface.co/datasets/dh-unibe/image-text_gt-fraktur) @ `eebb83d4e297a9e5b1c86e2ec2b1a2a305d3ad13`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_fibeln`](https://huggingface.co/datasets/dh-unibe/image-text_fibeln) @ `552ae4dd2669193b614ba3f19c4c13088c87f0b3`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_dtgt`](https://huggingface.co/datasets/dh-unibe/image-text_dtgt) @ `3d91f02647680fa30cd1a659d1ec03f66b929628`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_dach-gt`](https://huggingface.co/datasets/dh-unibe/image-text_dach-gt) @ `ec1faf7d900cbc78609c4ad7db152bbbf8d21951`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100
- [`dh-unibe/image-text_weisthuemer`](https://huggingface.co/datasets/dh-unibe/image-text_weisthuemer) @ `bb92c7525f93ff1f7c3ae215ef8f8c6eff834922`
  - Training projects: —
  - Evaluation: a seeded page-level split of the training projects (`partition=0.9`, `seed=42`)
  - Page cap: 100

Materialized from that selection: **1,681** pages, **60,743** transcribed lines, **58,101** training samples.

## Hyperparameters

```yaml
base_model: microsoft/trocr-base-handwritten
epochs: 3
batch_size: 8
accumulate_grad_batches: 1
lrate: 5.0e-05
lr_scheduler: cosine
warmup_ratio: 0.1
weight_decay: 0.0
max_grad_norm: 1.0
optim: adamw_torch
gradient_checkpointing: true
max_new_tokens: 256
beam_size: 4
length_penalty: 1.0
no_repeat_ngram_size: 3
eval_samples: 500
seed: 42
workers: 8
device: cuda:0
wandb_run: null
precision: bf16
```

## Provenance

| | |
|---|---|
| engine | `trocr` |
| base model | [`dh-unibe/trocr-kurrent-XVI-XVII`](https://huggingface.co/dh-unibe/trocr-kurrent-XVI-XVII) |
| training job | `20261026T064540Z-trocr-german-kurrent-xvi-xix-from-kurrent-v2` |
| code | evaluated with `50f7c4fb0cea` |
| trained | 2026-10-26T09:54:27.466469+00:00 |
| weights | `config.json`, `generation_config.json`, `merges.txt`, `model.safetensors`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `training_args.bin`, `training_summary.json`, `vocab.json` |

`metadata.json` in this repo is the record the trainer wrote, verbatim: the full request, the parsed metrics and the job id.

## Energy and carbon

| | |
|---|---|
| energy at the cards | **0.443 kWh** (443 Wh) |
| CO2e, market-based | **9 g** at 20 g/kWh |
| CO2e, location-based | 51 g at 115 g/kWh (Swiss consumer mix) |

Since 2016 the university has procured electricity from 100 % renewable sources (hydro, biomass, solar), predominantly Swiss ([source](https://klima.unibe.ch/betrieb/energie/index_ger.html)), which is the market-based factor. The location-based figure is what the local grid mix would have cost and is reported beside it, as the GHG Protocol asks when a market instrument is used.

**This is a floor.** It excludes data-centre overhead (PUE unknown; typical values 1.1–1.6 would raise it by that factor); it counts GPU draw only — CPU, memory, storage and network are not metered; it counts operation only — manufacture of the hardware is not apportioned; it a card was shared with another process, so part of this energy is not this run's.

## Using it

Weights: `config.json` (engine: `trocr`).
