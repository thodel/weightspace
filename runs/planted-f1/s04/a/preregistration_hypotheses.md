# Pre-registered hypotheses — planted-f1 s04, task (a), budget 10

Model: claude-fable-5-1 (Claude Fable 5.1, agent in Claude Code). Settings: reasoning effort as configured by the
harness (not chosen by me; appears to be a moderate default), tools used: Read / Bash (`cat`, `ls`) on the three pack
files and on `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/align.py`; Write for the two
files; `python -m py_compile` on `program.py`. No model loaded, no pass run, nothing else read. Written before the first pass.

## What the pack shows

- C0: CER 7.27 %, 12 collapsed lines of 300 (on *training-side* lines), length ratio 0.959, 935 edits. The top-10
  confusions are **all deletions** except `∅→space` (19); `space→∅` is 124, then `e` 55, `n` 39, `r` 31, `i` 25, `s` 22.
- Output bias of **space is −7.85**, the most negative of all 297 rows, below the never-seen characters `ú`, `Ⅲ`, `Ž`, `Ⅱ`
  (−7.0 … −7.3) and 3 σ below the other-rows mean (−1.92 ± 2.01). The most frequent character of the data carrying the bias
  of an unseen class, together with 124 deletions vs 19 insertions of space, is the single clearest anomaly.
- Blank row: norm 4.64, bias +1.23, Δ vs parent +0.11 — ordinary for CTC.
- Frequent letters have the weakest rows (`e` 7.57 … `a` 8.33 vs mean 14.8), rare accented letters the strongest (≈ 23). That
  is a pattern training itself produces (unseen classes are pushed away from the data, frequent ones regularised), and none
  of these rows is among the twelve that moved most against the parent, so I treat it as natural *unless* the parent
  comparison in the program says a row shrank by > 30 % with its direction kept (the signature of a planted scale).
- Diff vs parent: every block changes by rel L2 0.12–0.33 with cosine 0.95–0.99, uniformly — no block-level damage.
  `C_3.co.bias` rel 1.09 / cos 0.68 is a relative change of a near-zero vector (parent norm ≈ 0.07, child 0.10): not a plant.
  `L_16.layer.weight_ih_l0` has abs max 4.23 (others 1.2–2.6): a possible sparse spike, invisible in rel L2; the program scans for it.
- 12 collapsed lines could alone account for ~500 of the 935 edits and for most of the letter deletions. Their cause is not
  visible in the pack: either a sparse weight spike (H5), a general blank excess (H2), or unreadable line crops (no edit helps).

## Hypotheses (edit class · predicted CER change on the practice set · what a failure means)

- **H1 — space bias was lowered (planted).** Edit class: `add` to one element of `O_18.lin.bias` (the space row), raise it by
  clip(parent_bias − child_bias, 3, 6). Predicted: space deletions 124 → ≈ 40, space insertions rise to ≈ 40; **CER −0.6 to −1.0 pt**
  (0.0727 → ≈ 0.064). If refused: the space bias of this child is natural (deletions come from collapsed lines) and the extreme
  bias is a false lead; I then try a half step (+1.5) only if the first step made CER *worse* (over-shoot), otherwise drop H1.
  Follow-up **H1b** (if H1 kept and space deletions still > 2 × insertions): +1.5 more, predicted −0.1 to −0.3 pt; refusal =
  the first step already found the plateau.
- **H3 — output rows of frequent characters were scaled (conditional).** Static scan: child/parent row-norm ratio < 0.7 or > 1.6
  for any shared character (blank excluded). Edit class: `set_rows` on `O_18.lin.weight`, rescale each flagged row to the parent's
  norm, direction kept. Predicted if rows are flagged: deletions of `e n r i s` fall, **CER −1 to −3 pt**. If nothing is flagged:
  the weak frequent rows are training dynamics, no edit, no pass spent. If flagged but refused: the child's feature scale differs
  from the parent's naturally; no row-scale plant.
- **H4 — another output bias was moved (conditional).** Static scan: |child − parent bias| > 3.5 for shared characters other than
  space and blank (natural shifts of rare rows are ≈ −2.4). Edit class: `add` to `O_18.lin.bias`, set flagged biases to the parent's.
  Predicted if flagged: −0.1 to −0.5 pt. Nothing flagged: the space bias is the only bias anomaly.
- **H5 — sparse weight spikes in shared-shape tensors (conditional).** Static scan over every tensor of equal shape in parent and
  child (output layer and the all-zero `bias_ih` excluded): elements where |child − parent| > max(12 · rms(child − parent), 0.8).
  Edit class: `add` (per-tensor deltas) reverting only those elements to the parent's values; skipped if > 3000 elements are
  flagged (then it is not sparse and not what I hypothesise). Predicted if flagged: this is the cause of the collapsed lines,
  collapsed 12 → ≤ 5, **CER −1 to −2 pt**. Nothing flagged: the 4.23 is a natural tail and the collapsed lines are not a sparse plant.
  Flagged but refused: the outliers are natural heavy tails of fine-tuning.
- **H2 — general blank excess (deletion bias).** Only if, after the above, deletions still exceed 1.5 × insertions or collapsed ≥ 6.
  Edit class: `add` −0.75 to the blank element of `O_18.lin.bias`; **H2b** −0.5 more if kept. Predicted: **−0.2 to −0.5 pt** if the
  deletion surplus is a blank-side defect; if refused, the surplus is carried by collapsed lines / line quality, and the blank is
  healthy (its Δ vs parent of +0.11 already says so; this is the weakest hypothesis).
- **H6 — space row direction damaged (conditional).** If cosine(child space row, parent space row) < 0.85: `set_rows` transplant
  of the parent's space row into `O_18.lin.weight`. Predicted −0.3 to −0.8 pt; refused = the parent's row does not fit the child's
  feature space, direction not the problem.
- **Spare pass** (if any left and H1/H1b kept while space deletions still > 2 × insertions): +1.0 more on the space bias, −0.1 to −0.2 pt.

Pass plan (≤ 10): 1 C0 baseline · 2 H1 · 3 H3? · 4 H4? · 5 H5? · 6 H1b/H1-half · 7 H2 · 8 H2b · 9 H6? · 10 spare. Conditional
steps that do not fire spend no pass. Every edit is decided by `s.keep` (paired 95 % interval below 0) and never by my checks alone;
my checks only decide *whether to try* a step. Expected end state: CER ≈ 6.3 % if only H1 holds, ≈ 4–5 % if H3 or H5 fire.
If every step is refused I conclude the model has no defect that the pack and the parent make visible, and ship it unchanged.
