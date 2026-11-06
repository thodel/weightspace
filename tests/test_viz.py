"""Engine-free checks of wsp.viz: no weights, no kraken — synthetic readings and the committed α-sweep JSON."""
import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from wsp import viz  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_tensor_order_matches_state_dict():
    names = ["L_12.layer.bias_hh_l0_reverse", "C_3.co.bias", "O_18.lin.weight", "L_12.layer.weight_ih_l0",
             "C_3.co.weight", "L_12.layer.weight_hh_l0_reverse", "C_0.co.weight", "L_12.layer.bias_hh_l0"]
    assert sorted(names, key=viz.tensor_key) == [
        "C_0.co.weight", "C_3.co.weight", "C_3.co.bias", "L_12.layer.weight_ih_l0", "L_12.layer.bias_hh_l0",
        "L_12.layer.weight_hh_l0_reverse", "L_12.layer.bias_hh_l0_reverse", "O_18.lin.weight"]
    assert viz.short_name("L_12.layer.weight_hh_l0_reverse") == "L_12 hh←"
    assert viz.short_name("C_3.co.bias") == "C_3 b"


def test_glyph_labels():
    assert viz.glyph("<blank>") == "blank" and viz.glyph(" ") == "␣"
    assert viz.glyph("ͤ") == "◌ͤ" and viz.glyph("ſ") == "ſ"


def test_drift_records_from_readings_json():
    d = json.loads((REPO / "docs/readings/diff_german-xix-v1__german-xix-blb-v1.json").read_text())
    recs = viz.drift_records(d)
    assert len(recs) == 34 and recs[0]["name"] == "C_0.co.weight" and recs[-1]["name"] == "O_18.lin.bias"
    assert sum(r["how"] == "zero" for r in recs) == 6               # kraken's untrained bias_ih
    out_w = next(r for r in recs if r["name"] == "O_18.lin.weight")
    assert out_w["how"] == "rowmean" and out_w["rel"] == pytest.approx(d["output_rows"]["mean_rel"])


def test_drift_records_prefer_aligned_output():
    d = {"tensors": {"O_18.lin.weight": {"shape_a": [3, 2], "shape_b": [4, 2]}},
         "aligned_output": {"O_18.lin.weight": {"rel_l2": 0.2, "cosine": 0.9}}}
    (r,) = viz.drift_records(d)
    assert r["how"] == "aligned" and r["rel"] == 0.2


def test_row_table_pairs_by_character_not_index():
    parent = {"output_layer": {"rows": [{"char": "<blank>", "norm": 3.0, "bias": 1.0},
                                        {"char": "a", "norm": 5.0, "bias": 0.0}, {"char": "b", "norm": 6.0, "bias": 0.5}]}}
    child = {"output_layer": {"rows": [{"char": "<blank>", "norm": 3.5, "bias": 1.1},
                                       {"char": "b", "norm": 7.0, "bias": 0.2}, {"char": "a", "norm": 5.0, "bias": 0.0},
                                       {"char": "z", "norm": 2.0, "bias": -1.0}]}}
    rows = {r["char"]: r for r in viz.row_table(parent, child, Counter("aab"))}
    assert rows["b"]["norm_parent"] == 6.0 and rows["b"]["bias_delta"] == pytest.approx(-0.3)
    assert rows["z"]["norm_parent"] is None and rows["z"]["count"] == 0
    assert rows["a"]["count"] == 2 and rows["<blank>"]["count"] is None


def test_alpha_figure_from_committed_json(tmp_path):
    inp = viz.Inputs(tmp_path / "no-models")
    paths = viz.figure_alpha(inp, tmp_path, "svg")
    assert [p.name for p in paths] == ["alpha-sweep-xix-blb-v1.svg", "alpha-sweep-xix-blb-v1-dark.svg"]
    svg = paths[0].read_text()
    assert svg.startswith("<?xml") and "<dc:date>" not in svg
    assert "docs/baselines/alpha-xix-blb-v1.json" in inp.sources and not inp.used


def test_drift_and_architecture_without_weights(tmp_path):
    inp = viz.Inputs(tmp_path / "no-models")              # forces the committed readings
    paths = viz.figure_architecture(inp, tmp_path, "png") + viz.figure_drift(inp, tmp_path, "png")
    assert {p.name for p in paths} >= {"architecture.png", "drift-bern-dark.png", "drift-xix.png"}
    assert all(p.stat().st_size > 10_000 for p in paths)
    assert viz.figure_rows(inp, tmp_path, "png") == [] and viz.figure_padding(inp, tmp_path, "png") == []
    readme = viz.write_readme(tmp_path, paths, inp, ["all", "--out", "x"])
    assert "no weights read" in readme.read_text()


def test_svg_is_deterministic(tmp_path):
    sweep = json.loads((REPO / "docs/baselines/alpha-xix-blb-v1.json").read_text())
    a = viz.render(viz.plot_alpha, sweep, tmp_path / "a", "f")[0].read_bytes()
    b = viz.render(viz.plot_alpha, sweep, tmp_path / "b", "f")[0].read_bytes()
    assert a == b


def test_known_digests_parses_table():
    d = viz.known_digests()
    assert d["kraken-german-xix-blb-v1"].startswith("48caacb4")
