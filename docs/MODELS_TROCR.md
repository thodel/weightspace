# TrOCR model inventory (state 11.11.2026, from the Hugging Face hub)

Compiled from the hub on 11 November 2026 with the account `thodel` (admin of `dh-unibe`). Sizes, tags and cards are as the hub reports them; nothing here was
downloaded or measured. Digests, revisions and CERs of our own re-reading come from `scripts/trocr_fetch.py` and the stage-0 session
(`docs/epics/EPIC_TROCR_TRACK.md`). The gateway column is from `gateway_models` on the same day.

## Architecture (all base-size models; read from `config.json` of `trocr-german-kurrent-xvi-xix-from-kurrent-v2`)

`VisionEncoderDecoderModel`, 333.9 M parameters, float32 on disk (1.34 GB `model.safetensors`).

| part | what | numbers |
|---|---|---|
| encoder | ViT, patch 16, input **384 x 384** (the line crop is resized to a square), mean/std 0.5 | 12 layers, hidden 768, 12 heads, FFN 3072, no q/k/v bias |
| decoder | TrOCR text decoder, learned positions, `layernorm_embedding` | 12 layers, d_model 1024, 16 heads, FFN 4096, vocabulary **50,265** (RoBERTa byte-level BPE), cross-attention on the 768-d encoder output, 512 positions |
| tying | top-level `tie_word_embeddings: false`, but the decoder config ties | **corrected by stage 0 (docs/trocr/STAGE0.md): the hub files hold ONE embedding tensor (`decoder.model.decoder.embed_tokens.weight`, 50,265 x 1,024); the output projection is tied to it. Editing the one changes both input and output** |
| generation (trainer's eval) | beam 4, `max_new_tokens` 256, `no_repeat_ngram_size` 3, length penalty 1.0 | the repeat constraint changes lines that legitimately repeat a trigram; pin it per session |

Counted by stage 0: encoder 86,062,848 (without the unused ViT pooler of 590,592), decoder 247,268,352 (of which embedding 51,999,744, cross-attention 44,089,344), 333,921,792 stored parameters in all. A **large** family exists next to it
(558 M: presumably a 24-layer ViT-L encoder with the same 12-layer decoder, to be read from its `config.json`), see below; its tensors do not fit the base-size family.

**Consequence for weight-space work:** every base-size model of the Bern family and most third-party fine-tunes share architecture and tokenizer (`vocab.json` 798,293
bytes and `merges.txt` 456,318 bytes are identical in `trocr-kurrent` and `trocr-german-kurrent-xvi-xix-from-kurrent-v2`). Tensors are therefore **aligned by name**, with no
codec alignment as in kraken, and a sibling or parent is a true donor (unlike the adapters, cosine 0.10). Verified by stage 0: vocabulary (50,265 entries), merge list and special tokens are identical in all seven fetched models (the files differ in serialisation only; docs/trocr/STAGE0.md).

## A. dh-unibe (our own), base-size, line level

| repo | public | base | training | recorded CER (own split, not comparable between rows) | gateway |
|---|---|---|---|---|---|
| `dh-unibe/trocr-kurrent` | yes | `microsoft/trocr-base-handwritten` | Kurrent 19th c.: 292,997 train / 7,513 eval / 15,817 test lines, 20 epochs | eval 2.83 %, test 2.66 % | `trocr-kurrent`, enabled, 1,500 MB |
| `dh-unibe/trocr-kurrent-XVI-XVII` | yes | `dh-unibe/trocr-kurrent` | Kurrent 16th to 18th c., Swiss-biased, about 1.58 M words, 20 epochs | eval 5.67 %, test 5.42 % | `trocr-kurrent-xvi-xvii`, enabled |
| `dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2` | yes | `trocr-kurrent-XVI-XVII` | 20 public datasets (Zurich protocols, kurrent-xix, Basel Grundbuch, Königsfelden, Fraktur, ...), 1,681 pages, 53,812 train samples, **3 epochs**, lr 5e-5, cosine, bf16, job `20261026T064540Z-...` | split CER 8.89 %, WER 24.71 % (500 lines) | not registered |
| `dh-unibe/trocr-german-kurrent-xvi-xix-v2` | **private** | **`microsoft/trocr-base-handwritten`** | the same data and recipe (53,812 samples, 3 epochs) | split CER 11.00 % | not registered |
| `dh-unibe/trocr-german-kurrent-xvi-xix-full-v1` | **private** | `trocr-kurrent-XVI-XVII` | the same dataset list, 450,379 train samples, **1 epoch** | split CER 10.38 % (19,258 chars) | not registered |
| `dh-unibe/trocr-medieval-escriptmask` | yes | `microsoft/trocr-base-handwritten` | charter and book scripts, CREMMA Medieval LAT, HIMANIS, Itinera Nova, Königsfelden; eScriptorium line masks; 20 epochs | eval 3.29 % | `trocr-medieval-escriptmask`, enabled |
| `dh-unibe/trocr-towerbooks` | yes | not stated (card empty) | not stated | none | not registered |
| `dh-unibe/trocr-hgb`, `dh-unibe/trocr-medieval` | **private**, 2023 | not stated | not stated | none | not registered |

Large family (558 M):

| repo | base | training | recorded CER | gateway |
|---|---|---|---|---|
| `dh-unibe/trocr-essoins-middle-latin` | `magistermilitum/tridis_HTR` | Anglicana court hand (Latin), 15,356 train lines, 14.2 of 20 epochs | eval 5.44 %, test 6.22 % (CERberus: lowercase 4.84 %, capitals 6.29 %) | `trocr-essoins-middle-latin`, enabled |

Lineage inside the family (base-size):

```
microsoft/trocr-base-handwritten
 ├─ dh-unibe/trocr-kurrent (Kurrent XIX, 20 epochs, 293 k lines)
 │   └─ dh-unibe/trocr-kurrent-XVI-XVII (20 epochs)
 │       ├─ ...-xvi-xix-from-kurrent-v2   (3 epochs, 54 k samples)     public
 │       └─ ...-xvi-xix-full-v1           (1 epoch, 450 k samples)     private
 ├─ dh-unibe/trocr-german-kurrent-xvi-xix-v2 (3 epochs, 54 k samples; same data as from-kurrent-v2)   private
 └─ dh-unibe/trocr-medieval-escriptmask
```

This gives, with no training of our own: fine-tune distances from 1 to 20 epochs and 54 k to 450 k samples; **two models trained on the same data from different parents**
(`from-kurrent-v2` and `v2`: the nearest thing to the kraken sibling pair `xix-blb-v1` / `xix-v1` and the donor setting of the Bern sessions); a three-step chain
(microsoft, kurrent, XVI-XVII) for additivity along a lineage.

## B. Third-party, relevant

| repo | what | size | licence | remark |
|---|---|---|---|---|
| `microsoft/trocr-base-handwritten`, `-large-handwritten`, `-small-handwritten` | the English IAM-trained originals (3.3 M to 31 M downloads) | 333 M / large / small | MIT | the root of the family; the null for "what does fine-tuning change" |
| `microsoft/trocr-base-printed`, `-large-printed`, `-small-printed`, `-base-stage1`, `-large-stage1`, `-base-str` | printed, pre-training (stage 1) and scene text variants | | MIT | `stage1` is the pre-trained root before the handwritten fine-tune: one more step down the lineage |
| `Riksarkivet/trocr-base-handwritten-hist-swe-2` | Swedish historical handwriting, from `microsoft/trocr-base-handwritten`, 11 datasets | 384.9 M | Apache-2.0 | **stage-1 decoder architecture** (sinusoidal positions, no embedding norm, scaled embeddings, relu, untied output: +51 M), same 50,265 vocabulary. Weights are nearer `microsoft/trocr-base-stage1` than `base` (stage 1 readings, 12.11.2026): its card's parent is not what the weights show. Decoder blocks cannot be exchanged with the Bern models |
| `medieval-data/trocr-medieval-base` | CATMuS Medieval (la, fr, es) | 333.9 M | MIT | the TrOCR counterpart of the CATMuS kraken models; the incunabula-exchange analogue |
| `LaMOP/TrOCR_Manicule_2026_Latin_Medieval` and `..._Large_...` | Tridis + MEDUSA synthetic lines, from `microsoft/trocr-base-handwritten` and `-large-handwritten` | base / 558 M | MIT | 2026, two sizes from one recipe. **The base-size weights are unrelated to every model of the family** (distance 0.86 to 0.90, decoder 1.2; stage 1 readings, 12.11.2026), whatever the card says about its parent |
| `magistermilitum/tridis_HTR` | Latin/French/Spanish, 558 M, from `magistermilitum/Roberta_Historical` | 558.2 M | MIT | the parent of `essoins`; different decoder initialisation |
| `fhswf/TrOCR_german_handwritten`, `EpsilonGreedy/TrOCR_small_german_handwritten` | modern German handwriting (`fhswf/german_handwriting`) | 558 M / small | AFL-3.0 | out-of-domain German controls |
| `fgho/trocr-hanseXVI-kurrent`, `fgho/trocr-hanseXVII-kurrent` | Low German Kurrent, 16th and 17th c. | 333.9 M | none stated | no card; **weights show a fine-tune of our `trocr-kurrent-XVI-XVII`** (relative distance 0.018, bit-identical pooler), not of the base (stage 1 readings, 12.11.2026) |
| `slawa188/trocr-kurrent-galicia-lora` | a LoRA adapter on a TrOCR (Kurrent, Galicia) | adapter | not stated | the only TrOCR adapter seen; check what it adapts |
| `kazars24/trocr-base-handwritten-ru`, `cyrillic-trocr/trocr-handwritten-cyrillic`, `agomberto/trocr-*-fr`, `ddobokki/ko-trocr`, `team-lucid/trocr-small-korean`, `openthaigpt/thai-trocr`, `raxtemur/trocr-base-ru` | other languages, mostly fine-tunes of the English handwritten model | | MIT / Apache-2.0 | "fine-tune distance versus script distance" points; not for sessions |
| `Xenova/*`, `optimum-*`, `qualcomm/TrOCR` | ONNX or quantised conversions | | | not weights to study; skip |

Not covered by the search: models that do not carry `trocr` in the name or tags (for example Transkribus-style or DAN, DONUT, PaliGemma-type models, Pero), and
private repositories of other organisations.

## What this means for the track

- **Primary subject:** `dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2` (public, clean parent `trocr-kurrent-XVI-XVII`, same-data sibling `...-v2` and the large-data sibling `...-full-v1`).
- **Held-out lines** shared with the kraken Kurrent model: `data/heldout/blb-eval-hand` (2,272 lines, 29 objects, kraken `xix-blb-v1` 9.55 %). None of the 20 datasets in the card is a Laßberg/BLB
  project, but a document-level overlap audit with `wsp.audit` is still required (the Laßberg letters may sit in `kurrent-xix`'s Transkribus projects under another name).
- **Practice lines:** the BLB train side is unseen by these TrOCR models (it is what the kraken model was trained on), so it satisfies the proposed practice-set rule for this subject.
- Disk: each base-size model is 1.34 GB; the 10 models of stage 1 are about 14 GB. **h1 has 5.5 GB free**: everything is fetched and read on asteraix, never on h1.
