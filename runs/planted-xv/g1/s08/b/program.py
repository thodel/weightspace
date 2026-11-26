"""Pack s08, task (b): budget-0 blind session with an unrelated donor.

The readings in pack/ show that output row 'e' in O_18.lin.weight has L2 norm 0.00,
causing 910 'e' -> blank deletions and 376 'e' -> other vowel/letter substitutions.
However, the only allowed donor (kraken-bern-hertwig-v2) is from a different lineage
with an orthogonal L_16 feature space (cosine ~ 0.000), so its 'e' output row cannot
be transplanted. At budget 0 (no measurement pass to tune a bias threshold or verify
a heuristic vector), any blind edit to row 'e' risks catastrophic CTC over-emission.
We therefore log our diagnosis and freeze without unverified edits.
"""
from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()
    cm = s.subject.codec_map()
    e_row = cm["e"]
    e_norm = float((T["O_18.lin.weight"][e_row] ** 2).sum() ** 0.5)
    e_bias = float(T["O_18.lin.bias"][e_row])
    s.note(
        f"Diagnosis: O_18.lin.weight row for 'e' (row {e_row}) has L2 norm {e_norm:.4f} "
        f"and bias {e_bias:.4f}. Donor kraken-bern-hertwig-v2 is from an unrelated lineage "
        f"(L_16 cosine ~ 0.000), so row transplant is invalid. At budget 0, abstaining from "
        f"unverified heuristic edits to avoid unmeasured CTC degradation."
    )
    s.freeze()


if __name__ == "__main__":
    main()
