"""wsp.atlas_ablate: page split, unit names, ablation hooks on a synthetic network, scoring and choices."""
import numpy as np
import pytest

from wsp import atlas_ablate as AB


def test_split_pages_is_disjoint_and_balanced():
    items = [{"key": f"proj/p{i % 9}/l{i}"} for i in range(90)]
    a, b = AB.split_pages(items)
    pa = {x["key"].split("/")[1] for x in a}; pb = {x["key"].split("/")[1] for x in b}
    assert not pa & pb and len(pa) == 5 and len(pb) == 4 and len(a) + len(b) == 90


def test_unit_names_round_trip():
    units = AB.unit_list({"C_0": 32, "C_3": 32, "C_6": 64, "C_9": 64, "L_12": 400, "L_14": 400, "L_16": 400})
    assert len(units) == 1392
    assert all(AB._parse(AB.unit_name(u)) == u for u in units)
    assert AB.unit_name(("L_14", 203)) == "L_14 rev 3"


def _net():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.C_0 = nn.Conv2d(1, 3, (3, 3), padding=1)
            self.O_18 = nn.Conv2d(3, 3, (8, 1))
            with torch.no_grad():
                self.C_0.weight.zero_(); self.C_0.bias.zero_()
                self.C_0.weight[1, 0, 1, 1] = 1.0           # channel 1 passes the ink through
                self.O_18.weight.zero_(); self.O_18.bias.zero_()
                self.O_18.weight[1, 1] = 1.0                # class 1 reads channel 1
                self.O_18.bias[0] = 0.5                     # blank wins where there is no ink

        def forward(self, x, lens):
            for m in self.children():
                x = m(x)
            return x, lens

    return Seq().eval()


def test_ablation_hook_removes_what_the_unit_carries():
    net = _net()
    img = np.zeros((8, 12), np.float32); img[:, 3] = 1.0; img[:, 8] = 1.0
    vals = {"C_0": np.zeros(3, np.float32)}
    got = AB.read_ablated(net, img, [[], [("C_0", 0)], [("C_0", 1)], [("C_0", 1), ("C_0", 2)]], vals, {1: "a"}, batch=3)
    assert got[0] == "aa" and got[1] == "aa"           # ablating a silent channel changes nothing
    assert got[2] == "" and got[3] == ""               # ablating the ink channel removes both characters


def test_scorer_and_choose_pick_the_harmful_unit():
    refs = ["ab a", "ba b"]
    sc = AB.Scorer(refs, ["p1", "p2"])
    base = ["ab a", "ba b"]
    units = [("C_0", 0), ("C_0", 1)]
    stage_a = {"C_0 ch 0": base, "C_0 ch 1": ["b a", "b b"]}     # unit 1 deletes the a's
    ch = AB.choose(stage_a, units, sc, base)
    assert ch["top"][0] == "C_0 ch 1" and ch["dcer"]["C_0 ch 0"] == 0
    assert ch["circuits"]["a"] == ["C_0 ch 1"] and "C_0 ch 0" not in ch["loss"]["a"]


def test_loss_ci_point_and_interval():
    base = np.array([[5, 5], [4, 5], [5, 5], [3, 5]], float)
    abl = np.array([[2, 5], [1, 5], [4, 5], [3, 5]], float)
    r = AB.loss_ci(base, abl, ["p1", "p1", "p2", "p2"], resamples=200)
    assert r["loss"] == pytest.approx(100 * (17 - 10) / 20)
    assert r["ci95_cluster"][0] <= r["loss"] <= r["ci95_cluster"][1]


def test_weight_change_per_unit():
    rng = np.random.default_rng(0)
    H = 200                                            # unit names assume 200 units per direction
    shape = {"weight_ih": (4 * H, 3), "weight_hh": (4 * H, H), "bias_ih": (4 * H,), "bias_hh": (4 * H,)}
    par = {f"L_12.layer.{k}_l0{s}": rng.normal(size=shape[k]) for k in shape for s in ("", "_reverse")}
    chi = {k: v.copy() for k, v in par.items()}
    chi["L_12.layer.weight_ih_l0_reverse"][[1, H + 1, 2 * H + 1, 3 * H + 1]] += 1.0   # reverse unit 1 only
    old = AB.BLOCKS
    AB.BLOCKS = ["L_12"]
    try:
        wc = AB.weight_change(chi, par)
    finally:
        AB.BLOCKS = old
    assert len(wc) == 400 and wc["L_12 rev 1"] > 0
    assert sum(v > 0 for v in wc.values()) == 1
