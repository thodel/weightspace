from wsp.run import open_session


def main():
    s = open_session(".")
    h = s.hypothesis(
        "Several nearly parallel parent/child output rows have a common twofold norm gain; the head appears temperature-scaled.",
        "Softer posteriors; approximately zero percentage points change in greedy CER.",
        "joint output affine rescaling",
    )
    s.edit("scale", ["O_18.lin.weight", "O_18.lin.bias"], hypothesis=h, factor=0.5)
    s.keep_unmeasured("Undo the inferred common gain while preserving all framewise class orderings.")
    s.freeze()


if __name__ == "__main__":
    main()
