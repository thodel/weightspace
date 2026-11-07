# Brief for the asteraix session: a read-only weight survey of the kraken model zoo

Written 07.11.2026 on h1 for the manuscript (`paper/manuscript.md`, Section 6.5). Not started.

## Why

Section 6 of the article compares seven in-house models. The article's narrative asks for a comparison
of models from the kraken model zoo. This survey asks whether the three observations of
docs/READINGS.md hold beyond one project: (1) fine-tuning inflates the output rows of rarely seen
characters, (2) the blank bias rises in every fine-tune, (3) the convolutional stack moves less than the
recurrent stack and the output layer. It also asks how many public models use the standard kraken
architecture and whether lineage can be recovered from the weights alone.

## Material

- The 42 third-party models that the gateway lists in `serving-atr-inference/config/models.yaml`
  (Zenodo: CATMuS, TRIDIS, CREMMA, FoNDUE, OpenITI and others), plus the seven in-house models and
  the parent `cursive_inzigkofen` already under `models/kraken/`.
- Obtain each third-party file from its Zenodo record (DOI in the YAML), with `htrmopo` as in
  `deploy/asteraix_bootstrap.sh`, into `models/zoo/` (git-ignored). Record file sha256 and DOI.

## Rules

Everything of docs/BRIEF_PHASE0.md section 0 still applies: CPU only, card 1 is the trainer's, no write
to the share, the registry, HF or the trainer, `TMPDIR=~/atr-cache/tmp`, kraken 7.0.2. **No forward pass
is needed.** Weights are not committed; the survey tables and figures are.

## Tasks

1. Per model: format, VGSL specification, number of tensors and parameters, codec size, share of the
   codec that is Latin letters, digits, punctuation, combining marks and other; metadata keys that name a
   base model or training data.
2. Output layer (`O_*`): row norm distribution, the blank row's norm and bias, the rows of the ten most
   frequent letters against the rest. Use `wsp.stats.describe`.
3. Group models by identical VGSL specification. Report how many use the standard stack of the in-house
   models (`Cr3,13,32 … Lbx200 … O1c<N>`), which near-variants exist, and which are outliers.
4. Within a group, pairwise relative L2 and cosine per tensor, with output rows aligned by character
   (`wsp.align.codec_row_map`), never by index. Cluster the models; compare the clusters with the
   documented bases (metadata, Zenodo descriptions). Report where the weights agree with the record and
   where they do not. A model whose weights are almost identical to another is a result (a re-upload or
   a continuation).
5. For every documented parent-child pair in the group, repeat the three observations of the readings
   (rare-row inflation, blank bias, conv versus recurrent) and tabulate. State clearly when a pair is not
   a fine-tune of the other.
6. Outputs: `docs/ZOO_SURVEY.md` (method, tables, the three observations with counts, outliers, open
   questions), `docs/zoo/survey.json` and figures from `wsp.viz` in the same style as the existing ones.
   Add a short "what this means for Section 6.5 of the manuscript" paragraph, and no claim that the
   tables do not support.

## Not part of this task

No edits, no sessions, no scoring on held-out sets, no download of ground truth. If a model cannot be
loaded under kraken 7.0.2, list it with the error and go on.
