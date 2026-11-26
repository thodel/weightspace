from wsp.run import open_session


def main():
    s = open_session(".")
    tensors = [
        "L_16.layer." + param + "_l0" + suffix
        for suffix in ("", "_reverse")
        for param in ("weight_ih", "weight_hh", "bias_hh")
    ]
    h = s.hypothesis(
        "All nonzero L_16 parameters appear attenuated by a common factor of 0.5.",
        "Doubling them should reduce deletions and lower CER by 10-16 percentage points.",
        "scale",
    )
    s.edit("scale", tensors, hypothesis=h, factor=2.0)
    s.keep_unmeasured("Matching norm deficits in both directions support inverting the gain without replacing learned features.")
    s.freeze()


if __name__ == "__main__":
    main()
