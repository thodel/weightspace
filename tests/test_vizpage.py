"""wsp.vizpage builds from committed JSON alone; the embedded data is consistent and cannot close the script tag."""
import json
from pathlib import Path

from wsp import vizpage as vp

REPO = Path(__file__).resolve().parent.parent


def _probes():
    lay = {l: "10" for l in vp.LAYERS + ["model"]}
    return {"model": "m", "set": "s", "deleted_lost_at": {"never": 1},
            "lines": [{"key": "a/1", "ref": "x<", "hyp": "x", "status": ["ok", "del"], "runs": [(0, 0), (2, 3)],
                       "identity": lay, "emission": lay}]}


def test_page_data_joins_overlay_by_key_and_reference():
    ov = {"blocks": ["L_16"], "alphas": [0.0], "parent": "p",
          "lines": [{"key": "a/1", "ref": "x<", "states": {"L_16@0": {"hyp": "", "status": "dd"}}}]}
    d = vp.page_data(_probes(), ov)
    ln = d["lines"][0]
    assert d["tags"] == ["L_16@0"] and ln["ov"] == ["dd"] and ln["s"] == "od" and ln["x"] == [0.5, 3.0]
    assert ln["lost"] == ["", "never"]                        # the deleted second character: no layer has it
    ov["lines"][0]["ref"] = "other"
    assert vp.page_data(_probes(), ov)["lines"][0]["ov"] is None   # never pair by key alone


def test_render_page_is_self_contained():
    p = _probes()
    p["lines"][0].update(ref="</script>", status=["ok"] * 9, runs=[(i, i) for i in range(9)])
    p["lines"][0]["identity"] = p["lines"][0]["emission"] = {l: "0" * 9 for l in vp.LAYERS + ["model"]}
    page = vp.render_page(vp.page_data(p, None), [("probe-m__s", "caption")])
    assert page.count("</script>") == 2 and "probe-m__s-dark.svg" in page
    blob = page.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(blob)["lines"][0]["r"] == "</script>"


def test_committed_page_matches_committed_json():
    page = (REPO / "docs/viz/kraken-german-xix-blb-v1.html").read_text()
    data = json.loads(page.split('type="application/json">')[1].split("</script>")[0])
    probes = json.loads((REPO / "docs/viz/kraken-german-xix-blb-v1__blb-train-300.json").read_text())
    assert len(data["lines"]) == len(probes["lines"]) and all(ln["ov"] is not None for ln in data["lines"])
