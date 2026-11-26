Model: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API via operator tool layer).
Settings: provider default reasoning settings; tools: read_file, list_dir, write_file (py_compile), finish_round. Zero evaluation passes.

**Hypothesis:** The lowercase `e` classifier lost its feature weights. Its row norm rounds to zero, whereas even the blank has norm 4.64. Deleting `e` alone accounts for 910 of 2,088 edits; the substitutions for `e` fit loss of class evidence better than a globally excessive blank score.

**Ship one edit:** Replace only that weight row if effectively zero. Estimate its direction in the subject’s own coordinates from accented forms of `e`, subtracting accent effects estimated on other letters. Also consider the donor's *relations* between character rows, mapped through intact corresponding rows. Weight candidate reconstructions by how well they predict held-out intact lowercase rows from weights alone; a raw donor direction is credible only if it predicts intact rows. Calibrate norm against common subject letters. Leave bias, blank and feature extractor unchanged.

**Prediction:** A partial recovery of `e`, roughly 0.03–0.07 absolute CER reduction from 0.1623, predominantly fewer deletions. Accented variants need not share all the handwriting features of plain `e`; I would not expect an exact recovery or infer a second fault from residual errors.
