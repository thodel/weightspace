"""wsp.atlas on a small synthetic network: receptive field from the spec and measured, occlusion peaks, Gate 1 logic."""
import json

import numpy as np
import pytest

from wsp import atlas as A

XIX_SPEC = ("[1,120,0,1 Cr{C_0}3,13,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}3,13,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 "
            "Cr{C_6}3,9,64 Do{Do_7}0.1,2 Mp{Mp_8}2,2 Cr{C_9}3,9,64 Do{Do_10}0.1,2 S{S_11}1(1x0)1,3 Lbx{L_12}200 "
            "Do{Do_13}0.1,2 Lbx{L_14}200 Do{Do_15}0.1,2 Lbx{L_16}200 Do{Do_17} O{O_18}1c297]")


def test_spec_rf_gives_the_140_px_of_the_epic():
    rf = A.spec_rf(XIX_SPEC)
    assert [rf[b]["rf"] for b in ("C_0", "C_3", "C_6", "C_9")] == [13, 38, 72, 140]
    assert [rf[b]["jump"] for b in ("C_0", "C_3", "C_6", "C_9")] == [1, 2, 4, 8]
    assert rf["L_12"]["rf"] is None and rf["O_18"]["jump"] == 8


def test_windows_cover_the_line_and_clip():
    w = A.windows(40, 16, 4)
    assert w[0] == (0, 8, 4.0) and w[-1] == (32, 40, 36.0)
    assert all(0 <= lo < hi <= 40 for lo, hi, _ in w)


def test_sample_lines_skips_lines_the_model_saw_and_short_lines():
    items = [{"key": f"p/{i}/l", "text": "x" * 20, "seen_by": ["m"] if i % 2 else []} for i in range(10)]
    items.append({"key": "p/x/l", "text": "short", "seen_by": []})
    got = A.sample_lines(items, "m", 3)
    assert len(got) == 5 and all(not x["seen_by"] for x in got)
    assert [x["key"] for x in got] == [x["key"] for x in A.sample_lines(list(reversed(items)), "m", 3)]


def _net():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.C_0 = nn.Conv2d(1, 2, (3, 5), padding=(1, 2))
            self.Mp_1 = nn.MaxPool2d(2)
            self.C_3 = nn.Conv2d(2, 2, (3, 3), padding=(1, 1))
            self.Mp_2 = nn.MaxPool2d(2)
            self.O_18 = nn.Conv2d(2, 3, (4, 1))        # height 16 → 4 after two pools; collapses it
            with torch.no_grad():
                for m in (self.C_0, self.C_3):
                    m.weight.fill_(0.1); m.bias.zero_()
                self.O_18.weight.zero_(); self.O_18.bias.zero_()
                self.O_18.weight[1].fill_(2.0)        # class 1: ink
                self.O_18.bias[0] = 1.0              # blank: constant

        def forward(self, x, lens):
            for m in self.children():
                x = m(x)
            return x, lens

    return Seq().eval()


def test_occlusion_receptive_field_stays_inside_the_spec_bound():
    net = _net()
    rng = np.random.default_rng(0)
    img = (rng.random((16, 96)) > 0.7).astype(np.float32)
    r = A.occlude(net, img, ["C_0", "C_3", "O_18"], window=4, step=2)
    ppf = r["px_per_frame"]
    assert r["frames"] == 24 and ppf == 4
    rec = {"delta": r["delta"], "centres": r["centres"], "frames": r["frames"], "px_per_frame": ppf}
    spec = A.spec_rf("[1,16,0,1 Cr{C_0}3,5,2 Mp{Mp_1}2,2 Cr{C_3}3,3,2 Mp{Mp_2}2,2 O{O_18}1c3]")
    for b in ("C_0", "C_3"):
        p = A.rf_profile([rec], b, step=2)
        bound = A.rf_bound(spec[b]["rf"], spec[b]["jump"], ppf, window=4)
        assert 0 < p["extent"] <= bound + 2, (b, p["extent"], bound)


def test_occlusion_and_gradient_find_the_ink_of_a_character():
    net = _net()
    img = np.zeros((16, 96), np.float32)
    img[4:12, 40:48] = 1.0                            # one blob of ink around px 44
    r = A.occlude(net, img, ["C_0", "C_3", "O_18"], labels=[1], runs=[(10, 11)], window=8, step=2)
    peak = r["centres"][int(r["drop"][0].argmax())]
    assert abs(peak - 44) <= 4 and r["drop"][0].max() > A.MIN_DROP
    g = A.gradient(net, img, [1], [(10, 11)], window=8, step=2)
    assert abs(r["centres"][int(g[0].argmax())] - peak) <= 4


def _rec(peaks, centres_frames, status=None, ref=None, drop=1.0):
    xs = np.arange(0, 200, 4.0)
    n = len(peaks)
    d = np.zeros((n, len(xs)), np.float32)
    for i, p in enumerate(peaks):
        d[i, int(p // 4)] = drop
    return {"key": "proj/p1/l1", "ref": ref or "ab" * (n // 2) + "a" * (n % 2), "status": status or ["ok"] * n,
            "runs": [(f, f) for f in centres_frames], "px_per_frame": 8, "centres": xs, "drop": d}


def test_hit_rule_nearest_ctc_centre_and_min_drop():
    rec = _rec([4, 36, 60], [0, 4, 8])             # CTC centres 4, 36, 68 px
    rows = A.char_rows(rec)
    assert [r["hit"] for r in rows] == [True, True, True]    # 60: own centre 68 is 8 px away, the next (36) 24
    rec = _rec([4, 36, 52], [0, 4, 8])
    assert [r["hit"] for r in A.char_rows(rec)] == [True, True, False]   # 52: a tie (16 px to 36 and 68) is a miss
    assert not any(r["hit"] for r in A.char_rows(_rec([4, 36], [0, 4], drop=0.05)))


def test_gate_scores_ok_letters_only():
    rec = _rec([4, 40, 100, 140], [0, 4, 12, 17], status=["ok", "ok", "del", "ok"], ref="a bc")
    g = A.gate([rec])
    assert g["n"] == 2 and g["share"] == 1.0          # 'a' and 'c' scored; the space and the deleted 'b' are not
    assert g["not_gating"]["letter_del"]["n"] == 1 and g["not_gating"]["space_ok"]["n"] == 1


def test_cluster_share_interval_contains_point():
    rng = np.random.default_rng(1)
    hits = [(f"p{i % 7}", f"l{i % 30}", bool(rng.random() < 0.8)) for i in range(600)]
    c = A.cluster_share(hits, resamples=300)
    assert c["ci95_cluster"][0] <= c["share"] <= c["ci95_cluster"][1] and c["pages"] == 7


def test_page_is_self_contained_and_cannot_close_its_script():
    line = {"key": "k", "img": "img/k.png", "frames": 2, "ppf": 8, "width": 16, "ref": "</script>", "hyp": "",
            "status": ["del"] * 9, "runs": [[0, 0]] * 9, "centres": [4.0], "drop": [[0.0]] * 9, "grad": [[0.0]] * 9,
            "hit": [False] * 9, "peak": [4.0] * 9, "blank": [1, 1], "best": [0, 0], "bestc": "aa",
            "heat": {"C_0": A._q8(np.ones((2, 3)))}}
    page = A.render_page({"model": "m", "set": "s", "located": False, "lines": [line]})
    assert page.count("</script>") == 2
    blob = page.split('id=data>')[1].split("</script>")[0]
    assert json.loads(blob)["lines"][0]["ref"] == "</script>"
