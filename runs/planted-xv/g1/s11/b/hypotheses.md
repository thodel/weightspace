# Hypotheses — s11 task (b) (budget 0, blind programmer)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file` (with `py_compile`), `finish_round` (available: `read_file`, `list_dir`, `write_file`, `finish_round`).

## Diagnosis from pack readings

1. **Isolated defect in `L_14` weight matrices (`weight_ih_l0`, `weight_hh_l0`, `weight_ih_l0_reverse`, `weight_hh_l0_reverse`):**
   - In `diff_parent_to_subject.md`, the adjacent BiLSTM blocks `L_12` and `L_16` show normal fine-tuning drift from the parent (`kraken-german-xix-v1`):
     - `L_12`: `weight_hh` rel L2 `0.2079` / `0.2111` (cosine `0.9788` / `0.9781`), `weight_ih` rel L2 `0.2980` / `0.2946` (cosine `0.9556` / `0.9567`), `bias_hh` rel L2 `0.0419` / `0.0414` (cosine `0.9992`).
     - `L_16`: `weight_hh` rel L2 `0.1985` / `0.2188` (cosine `0.9806` / `0.9763`), `weight_ih` rel L2 `0.2964` / `0.3267` (cosine `0.9563` / `0.9467`), `bias_hh` rel L2 `0.0461` / `0.0441` (cosine `0.9990`).
   - In `L_14`, the biases `L_14.layer.bias_hh_l0` and `L_14.layer.bias_hh_l0_reverse` are completely healthy (rel L2 `0.0327` and `0.0317`, cosine `0.99951`), as are all convolutional blocks (`C_0`..`C_9`) and the output layer (`O_18`, 225 shared rows mean rel L2 `0.1930`, blank rel L2 `0.2253`, plus 72 new child codec rows).
   - However, the four weight matrices of `L_14` show anomalous relative L2 changes, reduced cosines, and inflated L2 norms in `describe.md`:
     - `L_14.layer.weight_hh_l0` (`[800, 200]`, $N = 160{,}000$): L2 `137.77`, rel L2 `0.5483`, cosine `0.87438`, abs max `2.063`.
     - `L_14.layer.weight_hh_l0_reverse` (`[800, 200]`, $N = 160{,}000$): L2 `133.93`, rel L2 `0.5572`, cosine `0.87110`, abs max `1.875`.
     - `L_14.layer.weight_ih_l0` (`[800, 400]`, $N = 320{,}000$): L2 `148.87`, rel L2 `0.7520`, cosine `0.78628`, abs max `1.863`.
     - `L_14.layer.weight_ih_l0_reverse` (`[800, 400]`, $N = 320{,}000$): L2 `149.60`, rel L2 `0.7400`, cosine `0.79233`, abs max `1.429`.
   - Solving the law of cosines $r^2 = 1 + \rho^2 - 2\rho\cos(P, C)$ for $\rho = \|C\| / \|P\|$ yields parent norms $\|P\| = 121.96, 118.07, 122.53, 123.52$ and excess squared norms $\|C\|^2 - \|C_0\|^2 \approx 3600$ for both `[800, 200]` matrices and $\approx 7200$ for both `[800, 400]` matrices. This matches i.i.d. zero-mean additive noise of standard deviation $\sigma = 0.15$ ($\sqrt{3600 / 160000} = \sqrt{7200 / 320000} = 0.150$) injected into the four `L_14` weight tensors, degrading baseline practice CER to `0.1200` (with 11 collapsed lines and generic character deletions).

## Shipped Edit (H1)

- **Hypothesis (H1):** The four weight matrices of the middle BiLSTM layer (`L_14.layer.weight_ih_l0`, `L_14.layer.weight_hh_l0`, `L_14.layer.weight_ih_l0_reverse`, `L_14.layer.weight_hh_l0_reverse`) are corrupted by additive isotropic noise ($\sigma \approx 0.15$), while `L_14` biases and all other blocks (`C_0`..`C_9`, `L_12`, `L_16`, `O_18`) are intact. Because `L_12`, `L_14`, and `L_16` share a tightly aligned hidden-state representation between parent and child (bias cosines `0.9990`–`0.9995`, uncorrupted weight cosines `0.95`–`0.98`), transplanting the four clean `L_14` weight matrices from the parent donor (`kraken-german-xix-v1`) eliminates the planted noise while preserving all child adaptations in the convolutional front-end, `L_12`, `L_14` biases, `L_16`, and the 297-class `O_18` output layer.
- **Edit class & operation:** `transplant` on `["L_14.layer.weight_ih_l0", "L_14.layer.weight_hh_l0", "L_14.layer.weight_ih_l0_reverse", "L_14.layer.weight_hh_l0_reverse"]` using donor tensors from `/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel` via `per_tensor`.
- **Predicted effect:** Practice CER (`0.1200` at C0) will decrease substantially by approximately `0.04` to `0.06` (down to ~`0.06`–`0.08`), collapsed lines will drop from 11 toward 0, and deletion errors (` `, `e`, `n`, `r`, `i` → `∅`) will be sharply reduced.
