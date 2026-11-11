# Pre-registration — 20261111-planted-k10-noise-lstm14-60pct-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k10-noise-lstm14-60pct-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s11/a/program.py`, copied unchanged.

1. **Subject:** `k10-noise-lstm14-60pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 60 % of the block's; `runs/planted/plant.json`) — tensor digest `930c5c2b0279e885f4e87954711d76aaee3e9d07128fc217156821da237eb2fc`, file sha256 `c55386f265d20e1ba0952391b63ff5e071a8277ec4544cbe9b0a86e9453f2bad`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Pre-registered hypotheses — planted-f1 / s11 / task (a), budget 10

Model: claude-fable-5-1 (Fable 5.1). Settings: reasoning effort as configured by the harness (not visible to me);
tools used: Read (pack, brief, the five allowed `wsp/` files), Bash (`ls` of my own folder, `py_compile`), Write.
No model loaded, program not run. Written 11.11.2026 before the first pass; `program.py` is fixed from here on.

## Readings that drive the hypotheses

- **L_14 block is displaced far more than its neighbours.** `weight_ih` (both directions) rel L2 0.75 / cos 0.79 and
  `weight_hh` 0.55 / cos 0.87 against the parent, where L_12 and L_16 sit at 0.30 / 0.96 (ih) and 0.21 / 0.98 (hh).
  L_14's biases moved *normally* (0.033, less than L_12/L_16). The child's L_14 norms are ~21 % (ih) and ~13 % (hh)
  above the parent's, while L_12/L_16 keep the parent's norm (natural drift is near-orthogonal and norm-preserving).
  Removing the natural-drift share (0.30 / 0.21) leaves an excess of 0.69 (ih) and 0.51 (hh) that corresponds to the
  *same* per-entry std of ≈ 0.15 on both shapes (800×400 and 800×200). That is the signature of i.i.d. noise added
  to the four L_14 weight matrices, not of training.
- **Space has the most negative output bias of all 297 classes (−7.85)**, below rare, never-seen classes (`ú`, `Ⅲ`, `Ž`
  −7.0 to −7.3), while frequent letters sit at ≈ 0 (`s` +0.40). ` ` → ∅ is the top confusion (156, 3× the next);
  ∅ → ` ` only 21. Length ratio 0.955, deletions dominate. A bias-only change on one row is invisible in the
  `moved most` table (ranked by weight distance), so the pack cannot confirm it; the program compares with the parent.
- `L_16.layer.weight_ih_l0` abs max 4.70, twice every other LSTM weight (1.4–2.4); tensor-level stats normal
  → if anything, a localized spike.
- 11 collapsed lines, deletions of e/n/r/i/s/t: consistent with an under-confident recurrent stack (noise in L_14).

Considered and **not** acted on: `C_3.co.bias` rel 1.09 / cos 0.68 is a tiny vector (parent L2 ≈ 0.07); its per-entry
drift (≈ 0.013) is in line with the other conv biases (0.008–0.027) — natural. Blank bias Δ +0.11, blank row rel 0.23
— natural. Rare punctuation rows (`«`, `¬`, `_`, `…`) pushed down by 2.4–2.7 with norms up — natural suppression of
classes absent from the child's data. 25 % of `C_0`/`C_3` weight entries unchanged from the parent — unexplained; only
actionable if they are exact zeros (checked in H4).

## Hypotheses (edit class; predicted CER change on the 300 practice lines; conclusion if it fails)

- **H1 — L_14 noise.** Interpolate the four `L_14.layer.weight_{ih,hh}_l0[_reverse]` toward the parent with a Wiener
  α = (var(Δ_L14) − var(Δ_L16)) / var(Δ_L14) computed in-program, clipped to [0.6, 0.95] (expected ≈ 0.85); biases untouched.
  Edit class: block interpolation to parent. Prediction: CER 0.120 → 0.06–0.09 (Δ −0.03 to −0.06), collapsed 11 → ≤ 5,
  deletions of e/n/r/i fall. If refused: one fallback at α = 0.4 (removes 40 % of a noise term while keeping most of the
  child's adaptation). If that is refused too: the L_14 displacement is the child's own training (or the parent's L_14 is
  incompatible with the child's L_12/L_16) and L_14 stays as it is; the noise reading was wrong.
- **H2 — planted output-row change, space first.** Scan all shared output rows against the parent (aligned by
  character): bias Δ beyond ±3.5 or weight rel L2 > 0.9 flags a row; also space alone if its bias Δ < −2. Repair flagged
  rows (1–10) with the parent's bias (and weights where the weight row is flagged). Edit class: output row
  transplant from parent. Prediction: Δ −0.005 to −0.012 (156 space deletions = 1.2 points at most). If the parent's
  space bias is also ≈ −7.85: nothing planted there; the −7.85 is natural (see H6). If refused: space deletions come
  from upstream, not from the bias; row stays.
- **H3 — localized LSTM damage in L_12 / L_16.** Entries with |w| > 3.0 and |w − parent| > 1.5 (spikes), or rows
  (gate units) with rel L2 > 1.2 or norm ratio outside [0.5, 2]: reset to the parent's values (one `add` edit, only if
  1–400 spikes / 1–40 rows). Edit class: sparse reset to parent. Prediction: Δ −0.001 to −0.01. If none found, no pass.
  If refused: the 4.70 is a natural outlier.
- **H4 — conv pruning / channel damage.** Conv weights with ≥ 5 % exact zeros where the parent is non-zero, or 1–8
  output channels with rel L2 > 0.9 or norm ratio outside [0.5, 2]: reset those entries/channels (+bias) to the parent.
  Edit class: sparse/channel reset to parent. Prediction: Δ −0.005 to −0.03 if triggered. If none found, no pass.
  If refused: the pattern is benign.
- **H5 — CTC calibration (not a defect repair).** If deletions still outnumber insertions > 1.5 : 1 after H1–H4:
  blank bias −0.5. Edit class: scalar bias shift. Prediction: Δ −0.002 to −0.008. A second −0.5 only if the first is
  kept and a pass remains. If refused: the deletion-heavy profile is the model's operating point, not a bias offset.
- **H6 — space calibration** (only if H2 found nothing planted on space): space bias +2.0. Edit class: scalar bias
  shift. Prediction: Δ −0.002 to −0.006. If refused: the −7.85 is natural and space suppression is upstream.

Pass plan: 1 baseline + H1 (1–2) + H2 (0–1) + H3 (0–1) + H4 (0–1) + H5 (0–2) + H6 (0–1) ≤ 10; every `measure` is
guarded by `budget_left`. Every edit is decided by `s.keep` (the harness rule) plus my own check that collapsed lines do
not rise by more than 5. Unspent passes stay unspent. Expected end state if H1 and H2 hold: CER ≈ 0.06–0.08.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s11/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
