"""Task (b) budget-0 repair program for pack s11.

Transplants the four noise-corrupted L_14 BiLSTM weight matrices from the parent
model (kraken-german-xix-v1), keeping L_14 biases and all other child blocks intact.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

L14_WEIGHTS = [
    "L_14.layer.weight_ih_l0",
    "L_14.layer.weight_hh_l0",
    "L_14.layer.weight_ih_l0_reverse",
    "L_14.layer.weight_hh_l0_reverse",
]


def main() -> None:
    s = open_session(".")
    parent = KrakenSubject(PARENT)
    pt = parent.tensors()

    h1 = s.hypothesis(
        "The four weight matrices of L_14 (weight_ih_l0, weight_hh_l0, weight_ih_l0_reverse, "
        "weight_hh_l0_reverse) are corrupted by additive isotropic noise (sigma ~ 0.15, inflating "
        "rel L2 vs parent to 0.55-0.75 and L2 norms by sqrt(N)*0.15 while L_14 biases remain at "
        "rel L2 0.032, cos 0.99951). Transplanting the four clean L_14 weight tensors from the "
        "parent donor removes the noise while preserving child adaptations in C_0..C_9, L_12, "
        "L_14 biases, L_16, and the 297-class O_18 output layer.",
        "CER decreases by ~0.04 to 0.06 from 0.1200 (to ~0.06-0.08) and collapsed lines drop from 11.",
        "transplant",
    )
    s.edit(
        "transplant",
        L14_WEIGHTS,
        hypothesis=h1,
        per_tensor={name: {"donor": pt[name]} for name in L14_WEIGHTS},
    )
    s.keep_unmeasured(
        "Transplant clean parent L_14 weight matrices to remove sigma=0.15 additive weight noise."
    )

    s.freeze()


if __name__ == "__main__":
    main()
