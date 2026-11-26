from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()
    cm = s.subject.codec_map()

    n_row = int(cm["n"])
    restored_row = (-T["O_18.lin.weight"][[n_row]]).copy()

    h1 = s.hypothesis(
        "Row 'n' of O_18.lin.weight is sign-flipped (cosine -0.991 vs parent, rel L2 2.042, norm 7.96 -> 8.33), causing 7,667 spurious 'n' insertions and missing true 'n'; negating row cm['n'] restores the healthy child weights.",
        "CER drops by ~0.65 to 0.70 (from 0.7456 to ~0.05-0.09)",
        "set_rows",
    )
    s.edit(
        "set_rows",
        ["O_18.lin.weight"],
        hypothesis=h1,
        index=[n_row],
        values=restored_row,
    )
    s.keep_unmeasured(
        "Restore negated O_18.lin.weight row for 'n' (cos -0.991 vs parent -> +0.991)."
    )

    s.freeze()


if __name__ == "__main__":
    main()
