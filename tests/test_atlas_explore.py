"""wsp.atlas_explore: per-frame scaling, a line on toy data, the model summary from the committed files, the page."""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from wsp import atlas as A
from wsp import atlas_explore as E

ATLAS = Path(__file__).resolve().parents[1] / "docs" / "atlas"


def test_cols_scale_each_frame_to_its_own_maximum():
    a = np.array([[1.0, 0.0, -4.0], [2.0, 0.0, 2.0]])
    q = E._cols(a)
    assert q[:, 0].tolist() == [128, 255] and q[:, 1].tolist() == [0, 0] and q[:, 2].tolist() == [255, 128]


def _rec(frames=10, windows=7):
    rng = np.random.default_rng(0)
    units = {"C_0": 32, "C_3": 32, "C_6": 64, "C_9": 64, "L_12": 400, "L_14": 400, "L_16": 400, "O_18": 5}
    lp = np.log(rng.dirichlet(np.ones(5), frames))
    return {"key": "doc/p1/l1", "frames": frames, "px_per_frame": 8, "width": frames * 8, "ref": "ab", "hyp": "ab",
            "status": ["ok", "ok"], "runs": [[2, 3], [6, 6]], "centres": np.linspace(0, frames * 8, windows),
            "logp": lp, "heat": {b: rng.normal(size=(frames, u)) for b, u in units.items()},
            "delta": {b: rng.random((windows, frames)) for b in units}}


def test_line_data_shapes_blank_and_window_step():
    rec = _rec()
    d = E.line_data(rec, {1: "a", 2: "b", 3: "c", 4: "d"})
    nw = len(range(0, 7, E.WIN_STEP))
    assert len(d["centres"]) == nw and set(d["heat"]) == set(E.LAYERS) and set(d["dep"]) == set(A.BLOCKS)
    assert len(base64.b64decode(d["dep"]["L_14"]["q"])) == rec["frames"] * nw            # frames × windows
    assert len(base64.b64decode(d["heat"]["L_14"]["q"])) == rec["frames"] * 400
    assert len(d["out"]) == rec["frames"] and all(len(o) == E.TOPK for o in d["out"])
    best = [o[0][0] for o in d["out"]]
    assert all(c in {"∅", "a", "b", "c", "d"} for c in best)
    assert d["img"] == "img/doc_p1_l1.png"


def test_js_payload_cannot_close_a_script():
    s = E._js("ATLAS_LINE", "m-0", {"ref": "a</script>b"})
    assert "</script>" not in s and json.loads(s.split("]=", 1)[1].rstrip(";\n"))["ref"] == "a</script>b"


def test_page_is_self_contained():
    page = E.render_page(["m1", "m2"])
    assert '["m1", "m2"]' in page and "__MODELS__" not in page
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|https?://", page)


@pytest.mark.skipif(not (ATLAS / "selectivity-kraken-bern-hertwig-v2__bern-schluss.json").exists(), reason="atlas outputs not built")
def test_model_data_from_committed_summaries():
    m, l, s, a = E.PAIRS[1]
    d = E.model_data(m, l, s, a, ATLAS)
    assert len(d["units"]) == 1392 and set(d["blocks"]) == set(A.BLOCKS)
    assert d["gate2"]["passes"] is False and d["blocks"]["L_14"]["probe"]["holds"] is True
    assert d["units"]["C_0 ch 3"]["u_claim"] is True and d["blocks"]["C_0"]["stability"]["parent"]["median_r"] > 0.9
    assert "probe" not in d["blocks"]["O_18"]
