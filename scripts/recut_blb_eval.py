"""Re-cut the crop-defect lines of blb-eval and score them with xix-blb-v1 (docs/PHASE4_REPORT.md §3).

    .venv/bin/python scripts/recut_blb_eval.py      # ~12 min on 12 CPU threads; writes docs/collapse/recut-blb-eval.json

Reads the classes of docs/collapse/blb-eval__xix-blb-v1.json (wsp.collapse). Crops are cut with a plain
rectangle from the page image (no extract_polygons) into ~/atr-cache/wsp/recut/:
- degenerate boxes (under 8 px high): the plain box ("plain"), and the box widened to the page's median
  line height (lines of 8 px or more) anchored at its top, middle or bottom ("deg_top/mid/bottom"), with
  15 % padding;
- vertical lines: the plain box and the box rotated 90 and 270 degrees (PIL: counter-clockwise);
- the six upside-down lines of K2915-3_0010: the plain box and the box rotated 180 degrees.
Nothing on the share is written.
"""
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from PIL import Image

from wsp.data import page_lines
from wsp.kraken_subject import KrakenSubject
from wsp.metrics import levenshtein, normalise, score

D = "/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt"
OUT = Path.home() / "atr-cache/wsp/recut"
UPSIDE = {f"manuskripte_eval/K2915-3_0010/r1l{n}" for n in (1, 3, 4, 5, 9, 11)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prof = json.loads(Path("docs/collapse/blb-eval__xix-blb-v1.json").read_text())
    cls = {x["key"]: x["class"] for x in prof["per_line"]}
    items = {json.loads(l)["key"]: json.loads(l) for l in open("data/heldout/blb-eval.jsonl")}
    base = {json.loads(l)["key"]: json.loads(l)["hyp"]
            for l in open("docs/baselines/readings/blb-eval/kraken-german-xix-blb-v1.jsonl")}
    cache = {}

    def page(proj, stem):
        if (proj, stem) not in cache:
            x = open(f"{D}/{proj}/page/{stem}.xml", encoding="utf-8").read()
            fn = re.search(r'imageFilename="([^"]+)"', x).group(1)
            ls = {l["id"]: l for l in page_lines(x)}
            hs = [max(p[1] for p in l["boundary"]) - min(p[1] for p in l["boundary"]) for l in ls.values()]
            hs = [h for h in hs if h >= 8]
            cache[(proj, stem)] = (Image.open(f"{D}/{proj}/{fn}").convert("L"), ls, st.median(hs) if hs else 60)
        return cache[(proj, stem)]

    variants = defaultdict(list)
    for k in sorted(set(cls) | UPSIDE):
        c = cls.get(k)
        if c not in ("degenerate_box", "vertical") and k not in UPSIDE:
            continue
        proj, stem, lid = k.split("/")
        im, ls, med = page(proj, stem)
        xs = [p[0] for p in ls[lid]["boundary"]]; ys = [p[1] for p in ls[lid]["boundary"]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        def save(name, crop):
            p = OUT / f"{name}__{k.replace('/', '_')}.png"; crop.save(p); variants[name].append((k, str(p)))
        if c == "degenerate_box":
            m = int(med); pad = int(0.15 * m)
            save("plain", im.crop((x0, y0, x1, max(y1, y0 + 1))))
            save("deg_top", im.crop((x0 - pad, y0 - pad, x1 + pad, y0 + m + pad)))
            save("deg_mid", im.crop((x0 - pad, y0 - m // 2 - pad, x1 + pad, y0 + m // 2 + pad)))
            save("deg_bottom", im.crop((x0 - pad, y0 - m - pad, x1 + pad, y1 + pad)))
        elif c == "vertical":
            cr = im.crop((x0, y0, x1, y1)); save("plain", cr)
            save("rot90", cr.rotate(90, expand=True)); save("rot270", cr.rotate(270, expand=True))
        if k in UPSIDE:
            cr = im.crop((x0, y0, x1, y1)); save("plain_up", cr); save("rot180", cr.rotate(180, expand=True))

    s = KrakenSubject("models/kraken/kraken-german-xix-blb-v1.mlmodel")
    hyps = {n: {p.key: p.text for p in s.forward([{"key": k, "image": f, "text": items[k]["text"]} for k, f in lst])}
            for n, lst in variants.items()}
    out = {"groups": {}}
    groups = {"degenerate_box": ("deg_top", ["plain", "deg_top", "deg_mid", "deg_bottom"]),
              "vertical": ("rot90", ["plain", "rot90", "rot270"]), "upside_down": ("rot180", ["plain_up", "rot180"])}
    for g, (anchor, names) in groups.items():
        keys = list(hyps[anchor]); refs = [items[k]["text"] for k in keys]
        row = {"lines": len(keys), "as_cut": score([base[k] for k in keys], refs).cer}
        for n in names:
            row[n] = score([hyps[n][k] for k in keys], refs).cer
        out["groups"][g] = row
        print(g, json.dumps(row))
    fixed = dict(base)
    for n in ("deg_mid", "rot90", "rot180"):
        fixed.update(hyps[n])
    allk = list(items); allr = [items[k]["text"] for k in allk]
    out["heldout_as_cut"] = score([base[k] for k in allk], allr).cer
    out["heldout_recut"] = score([fixed[k] for k in allk], allr).cer
    out["hyps"] = hyps
    Path("docs/collapse/recut-blb-eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("held-out CER as cut", out["heldout_as_cut"], "with the defect lines re-cut", out["heldout_recut"])


if __name__ == "__main__":
    main()
