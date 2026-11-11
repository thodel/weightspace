"""planted-f1 / s05 / task (b): blind programmer, budget 0.

One guarded edit (H1 of hypotheses.md): the third BiLSTM (L_16) looks scaled by ~0.5 against the
parent (bias_hh rel L2 0.505 at cosine 0.999; absolute norms half of L_14's). The program measures
the per-tensor least-squares scale of the subject against the parent -- no pass, the parent is only
a donor -- and, if every L_16 tensor sits near 0.5 while L_14 sits near 1.0, multiplies L_16 by the
common inverse factor. Otherwise nothing is applied.
"""
from __future__ import annotations

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

L16 = [f"L_16.layer.{t}{sfx}" for sfx in ("", "_reverse")
       for t in ("weight_ih_l0", "weight_hh_l0", "bias_hh_l0")]
L14 = [n.replace("L_16", "L_14") for n in L16]


def fit_scale(child: np.ndarray, parent: np.ndarray) -> tuple[float, float]:
    """Least-squares k with child ~ k * parent, and the cosine between them."""
    c = child.astype(np.float64).ravel()
    p = parent.astype(np.float64).ravel()
    pp = float(p @ p)
    k = float(c @ p) / pp if pp > 0 else float("nan")
    cos = float(c @ p) / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-12)
    return k, cos


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()
    P = KrakenSubject(PARENT).tensors()

    k16 = {n: fit_scale(T[n], P[n]) for n in L16}
    k14 = {n: fit_scale(T[n], P[n]) for n in L14}
    s.note("scale of subject vs parent, L_16: " +
           "; ".join(f"{n.split('.')[-1]}{'_rev' if n.endswith('_reverse') else ''}: "
                     f"k={k:.4f} cos={c:.4f} |child|={np.linalg.norm(T[n]):.2f} |parent|={np.linalg.norm(P[n]):.2f}"
                     for n, (k, c) in k16.items()))
    s.note("scale of subject vs parent, L_14 (control): " +
           "; ".join(f"{n.split('.')[-1]}{'_rev' if n.endswith('_reverse') else ''}: k={k:.4f} cos={c:.4f}"
                     for n, (k, c) in k14.items()))

    ks16 = np.array([k for k, _ in k16.values()])
    cos16 = np.array([c for _, c in k16.values()])
    ks14 = np.array([k for k, _ in k14.values()])
    guard = (bool(np.all((ks16 > 0.35) & (ks16 < 0.65))) and bool(np.all(cos16 >= 0.90))
             and bool(np.all((ks14 > 0.80) & (ks14 < 1.25))))

    if not guard:
        s.note("H1 guard failed: L_16 is not uniformly ~0.5x the parent while L_14 is ~1x. "
               "No edit applied (blind session; nothing else in the readings is a sure defect).")
        s.freeze()
        return

    factor = float(1.0 / np.median(ks16))
    h = s.hypothesis(
        "L_16 (third BiLSTM: weight_ih, weight_hh, bias_hh, both directions) was scaled by ~0.5 against the "
        "parent (bias_hh rel L2 0.505 at cos 0.999; norms half of L_14). Halved gate pre-activations squash the "
        "last LSTM's output, the logits lose contrast, the blank bias (+1.23 vs letters ~-1.9) wins -> deletions, "
        "length ratio 0.83, 24 collapsed lines. Undo by multiplying the six tensors by 1/median(k).",
        f"CER falls from 0.199 to about 0.05-0.10 (at least -0.08 absolute); deletion confusions shrink by most "
        f"of their count, collapsed lines -> ~0, length ratio -> ~1.0. Common factor applied: {factor:.4f} "
        f"(per-tensor k: {', '.join(f'{k:.3f}' for k in ks16)}).",
        "scale")
    s.edit("scale", L16, hypothesis=h, factor=factor)
    s.keep_unmeasured(
        f"budget 0: L_16 fitted at k={np.median(ks16):.4f} of the parent with cosines >= {cos16.min():.3f}, "
        f"L_14 control at k in [{ks14.min():.3f}, {ks14.max():.3f}]; undone by factor {factor:.4f}")
    s.freeze()


if __name__ == "__main__":
    main()
