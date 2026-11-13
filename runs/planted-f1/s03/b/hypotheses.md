# Hypotheses — planted-f1 / s03 / task (b), blind programmer (budget 0)

- Model: claude-fable-5-1 (Claude Fable 5.1), reasoning effort: default (medium), 11.11.2026.
- Tools used: Read / Bash `cat` on the three pack files and the allowed API files (`wsp/ops.py`, `wsp/session.py`,
  `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/align.py`), `py_compile` of `program.py`. No pass, no model loaded.

## What the readings say

- C0: CER 0.0789, WER 0.2135, **14 collapsed lines**, length ratio 0.941, 1015 edits. The top-10 confusions are all
  deletions (` ` 177, `e` 81, `n` 53, `i` 38, `r` 37, `s` 28, `t` 26, `.` 26, `o` 23, `u` 21); no substitution, no insertion.
- The non-space deletion counts match the letter frequencies of German text over ≈600 characters (16 % e, 10 % n, 7–8 % i/r,
  6 % s/t), which is what 14 collapsed lines of ≈43 characters contain. So ≈60 % of the edits sit in the 14 collapsed lines and
  the other 286 lines run at ≈3 % CER with ≈5 % of their spaces missed — a healthy-looking child. The question is only
  whether the collapse of 14 lines is a weight defect (input-dependent: a saturating scaled row/column) or the data (slivers,
  unreadable crops, see `KrakenSubject.forward`).
- Tensor-level diff to the parent is uniform and natural: rel L2 0.12–0.33, cosine 0.95–0.99 in every block, biases_hh
  rel 0.03–0.05, bias_ih zero in both. `C_3.co.bias` rel 1.09 / cos 0.68 is 0.11 in absolute L2 on a vector of rms 0.018 —
  the same absolute change as `C_9.co.bias`; negligible after a pre-activation of several units.
- Output rows aligned by character: mean rel 0.19, the most-moved rows (rare punctuation) have cos ≥ 0.86, norm ratio ≤ 1.36;
  blank row rel 0.23. No row of the output layer looks replaced or scaled.
- Oddities worth a data-driven check: `L_16.layer.weight_ih_l0` abs max 4.70 (every other weight tensor ≤ 2.84) and L2 136.9
  against 128.1 for its reverse twin (L_12/L_14 twins agree within 1 %); the space bias −7.85 is the minimum of all 297
  biases (below `Ⅲ`, `Ⅱ`, `Ž`); the blank bias is +3.23, +2.11 above the parent.

## Hypotheses (edit class, predicted effect, what each failing means)

H1 — a scaled or replaced row/column in an LSTM or conv tensor (candidate `L_16.layer.weight_ih_l0`), collapsing lines whose
    features drive that unit into saturation. Edit class: `set_rows` / `set_cols` after a per-row and per-column comparison
    with the parent (norm ratio normalised by the tensor's median ratio; cosine). A row is repaired only if its log norm ratio
    is > max(6 robust σ, ln 1.75) from the median (rescaled to the median ratio) or its cosine to the parent row is < 0.5 and
    8 robust σ below the median (parent row transplanted at the median ratio). Prediction if it fires: collapsed lines 14 → ≈0,
    CER −3 to −4.5 pp; if natural rows are caught instead (unlikely at these thresholds) ≤ +0.3 pp. If nothing fires, the
    collapses are the data or an undetectable plant (project_out), and no edit is made.
H2 — space bias planted low (−7.85, top confusion ` `→∅ 177). I think this is **natural**: outside the collapsed lines only ≈5 %
    of spaces are missed, whereas a 6-unit bias plant would delete nearly all of them; kraken whitespace biases are strongly
    negative in healthy models. Check: shift of the space bias from parent to child relative to the median shift of the
    frequent lower-case letters; repair (`add` on `O_18.lin.bias`, restore to the letters' median shift) only if < −3.5.
    Predicted if it fires: space deletions 177 → ≈80, CER −0.5 to −0.8 pp. Same rule for `.` `,` and each frequent letter.
H3 — blank bias planted high (+2.11 vs parent; deletions only). Probably natural drift: it is the same magnitude as the
    −2.4 drift of the rare-class biases, so the training moved biases by this much. Repair only if the blank's shift relative
    to the frequent letters is ≥ 3.5 (then lower it by shift − 1). Predicted if it fires: CER −1 to −3 pp, collapsed lines down.
H4 — a bias element plant (`C_*.co.bias`, `L_*.bias_hh`): element-wise child − parent outlier > max(10 robust σ, 0.3) → set
    back to parent + median shift (`add`). The readings show no such element (conv bias abs max ≤ 0.65); expected not to fire.
H5 — output rows/columns replaced or scaled (aligned by character): same detector as H1; the readings already exclude it.
    A rare-class bias pushed down by > max(10 σ, 4) is set back (harmless either way).

Nothing is applied unconditionally. If the readings and the parent comparison show nothing, the program freezes with notes only
("apply nothing if the readings show nothing wrong"). At most 10 edits; a tensor with > 30 % of its rows flagged is left alone
(the median is then not a reference).
