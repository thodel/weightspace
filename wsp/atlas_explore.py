"""Atlas explorer: one interactive page over the atlas results (epic #13), for a static web host.

A cursor moves along a text line; for the frame under it the page shows every unit of every block, the output, and
what part of the line the selected block's frame depends on (L2 occlusion). Clicking a block or a unit shows its
results with their controls (L1 selectivity, L3 ablation, seed stability). Nothing here is computed anew: the lines
are the viewer lines of `wsp.atlas` (`view_lines`), the numbers come from the committed summaries in `docs/atlas/`.

Output, next to the other atlas files:

- `explorer.html`: the page (no server, no external script; works from `file://`);
- `explore/<model>.js`: block and unit results of one model;
- `explore/<model>-<i>.js`: one line (activations, output, dependence), loaded when it is shown;
- `img/<key>.png`: the line images (`wsp.atlas.write_view_images`).

    python -m wsp.atlas_explore --out docs/atlas
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from wsp import atlas as A

PAIRS = [  # model, L0/L2 set (lines), L1 set (selectivity), L3 set (ablation)
    ("kraken-german-xix-blb-v1", "blb-val-300", "blb-val-300", "blb-val-300"),
    ("kraken-bern-hertwig-v2", "bern-val", "bern-schluss", "bern-val"),
]
LAYERS = [b for b in A.BLOCKS if b != "O_18"]
TOPK = 5                      # output characters shown per frame
WIN_STEP = 2                  # keep every 2nd occlusion window (8 px, one frame)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def _cols(a: np.ndarray) -> np.ndarray:
    """(windows, frames) → uint8 with every frame's column scaled to its own maximum (shape, not size)."""
    a = np.abs(np.asarray(a, np.float32))
    m = a.max(0, keepdims=True)
    return np.round(a / np.where(m > 0, m, 1) * 255).astype(np.uint8)


def line_data(rec: dict, l2c: dict[int, str]) -> dict:
    lp = np.asarray(rec["logp"], np.float64)
    p = np.exp(lp)
    top = np.argsort(-p, 1)[:, :TOPK]
    heat = {b: A._q8(rec["heat"][b]) for b in LAYERS}
    k = np.arange(0, len(rec["centres"]), WIN_STEP)
    dep = {b: {"q": _b64(_cols(rec["delta"][b][k]).T)} for b in A.BLOCKS}     # frames × windows
    return {
        "key": rec["key"], "img": f"img/{A._safe(rec['key'])}.png", "frames": rec["frames"], "ppf": rec["px_per_frame"],
        "width": rec["width"], "ref": rec["ref"], "hyp": rec["hyp"], "status": rec["status"], "runs": rec["runs"],
        "centres": np.asarray(rec["centres"])[k].round(1).tolist(),
        "out": [[["∅" if int(i) == 0 else l2c.get(int(i), "?"), round(float(p[t, i]), 3)] for i in top[t]]
                for t in range(len(top))],
        "heat": heat, "dep": dep}


def _cards(path: Path) -> dict:
    page = path.read_text()
    blob = page.split("id=data>")[1].split("</script>")[0]
    return {c["unit"]: c for c in json.loads(blob)["cards"]}


def _rnd(x, n=3):
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, list):
        return [_rnd(v, n) for v in x]
    if isinstance(x, dict):
        return {k: _rnd(v, n) for k, v in x.items()}
    return x


def model_data(model: str, lset: str, sset: str, aset: str, src: Path) -> dict:
    g = json.loads((src / f"atlas-{model}__{lset}.json").read_text())
    s = json.loads((src / f"selectivity-{model}__{sset}.json").read_text())
    a = json.loads((src / f"ablation-{model}__{aset}.json").read_text())
    cards = _cards(src / f"units-{model}__{sset}.html")
    blocks = {}
    for b in A.BLOCKS:
        rf = g["rf"][b]
        d = {"rf": {"spec": rf["spec_rf"], **{k: rf["trained"][k] for k in ("extent", "half_width_50", "half_width_90")}}}
        if b in LAYERS:
            d["probe"] = s["claims_p"][b]
            d["dirs"] = {h: s["claims_p"][f"{b} {h}"] for h in ("fwd", "rev") if f"{b} {h}" in s["claims_p"]}
            d["context"] = {t: s["prediction_d"][f"{b} {t}"] for t in ("prev", "next") if f"{b} {t}" in s["prediction_d"]}
            d["sel"] = s["blocks"][b]
            d["random_bar"] = s["random_top"][b]
            d["stability"] = (s.get("stability") or {}).get(b)
            d["group"] = a["claims"]["groups"].get(b)
            d["single"] = a["stage_a"].get(b)
            d["s_tested"] = [u for u, c in s["claims_s"].items() if c["block"] == b]
        blocks[b] = _rnd(d)
    circuits = {c: {"units": v["units"], "holds": v["holds"], "loss": v["loss"], "other": v["other_loss"]}
                for c, v in a["claims"]["circuits"].items()}
    member: dict[str, list[str]] = {}
    for c, v in circuits.items():
        for u in v["units"]:
            member.setdefault(u, []).append(c)
    units = {}
    for u, c in cards.items():
        units[u] = _rnd({"top": c["top"][:3], "auroc": c["auroc"], "best": c["best_selection"], "random": c["random"],
                         "blank": c["blank"], "shuffle": c["shuffle"], "dead": c["dead"], "seeds": c.get("seeds"),
                         "dcer": a["dcer_selection"].get(u), "circuits": member.get(u, []),
                         "s_claim": s["claims_s"].get(u, {}).get("verdict"),
                         "u_claim": a["claims"]["units"].get(u, {}).get("holds")}, 2)
    return {"model": model, "sets": {"lines": lset, "selectivity": sset, "ablation": aset},
            "gate1": _rnd(g["gate1"]), "gate2": s["gate2"], "blocks": blocks, "units": units,
            "circuits": _rnd(circuits, 1), "unit_claims": _rnd(a["claims"]["units"], 2)}


def _js(var: str, key: str, obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"window.{var}=window.{var}||{{}};window.{var}[{json.dumps(key)}]={blob};\n"


def build(out: Path, models_dir: Path = Path("models/kraken"), pairs=PAIRS) -> list[Path]:
    from wsp.kraken_subject import KrakenSubject
    out = Path(out)
    (out / "explore").mkdir(parents=True, exist_ok=True)
    written, index = [], []
    for model, lset, sset, aset in pairs:
        md = model_data(model, lset, sset, aset, out)
        recs = A.view_lines(A.load_set(model, lset))
        l2c = {v: k for k, v in KrakenSubject(models_dir / f"{model}.mlmodel").codec_map().items()}
        A.write_view_images(recs, out)
        md["lines"] = [{"ref": r["ref"], "frames": r["frames"]} for r in recs]
        for i, rec in enumerate(recs):
            p = out / "explore" / f"{model}-{i}.js"
            p.write_text(_js("ATLAS_LINE", f"{model}-{i}", line_data(rec, l2c)))
            written.append(p)
        p = out / "explore" / f"{model}.js"
        p.write_text(_js("ATLAS_MODEL", model, md))
        written.append(p)
        index.append(model)
        written += [out / "img" / f"{A._safe(r['key'])}.png" for r in recs]
    page = out / "explorer.html"
    page.write_text(render_page(index))
    return [page] + written


def render_page(models: list[str]) -> str:
    return Path(__file__).with_name("atlas_explore.html").read_text().replace("__MODELS__", json.dumps(models))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.atlas_explore")
    ap.add_argument("--out", default="docs/atlas")
    a = ap.parse_args(argv)
    for p in build(Path(a.out)):
        print(p)


if __name__ == "__main__":
    main()
