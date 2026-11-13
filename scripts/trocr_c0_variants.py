#!/usr/bin/env python3
"""Does a speed or batching setting move C0? Same model, same lines, generation with (a) the pinned settings, (b) use_cache False
(what the hub's generation_config.json says), (c) batch 1 (what the trainer's eval does), (d) the hub's length_penalty 2.0.
Reports lines whose text differs from (a) and the CER of each variant. Card 0 (checked), first N lines of the set."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("set"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
if a.device.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from wsp import card
    ok, why = card.ready(); print(why, flush=True)
    if not ok: sys.exit(3)
from wsp import data, metrics
from wsp.trocr_subject import TrocrSubject
rows = data.load(a.set)[:a.n]
refs = [r["text"] for r in rows]
res = {}
base = None
for name, kw, batch in [("pinned", {}, 8), ("use_cache_false", {"use_cache": False}, 8), ("batch_1", {}, 1),
                        ("hub_length_penalty_2", {"length_penalty": 2.0}, 8)]:
    s = TrocrSubject(a.model, device=a.device, batch=batch, generation=kw)
    t = time.time(); p = s.forward(rows); dt = time.time() - t
    texts = [x.text for x in p]
    if base is None: base = texts
    m = metrics.score(texts, refs, blocks=False)
    res[name] = {"cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "seconds": round(dt, 1),
                 "lines_differing_from_pinned": sum(x != y for x, y in zip(texts, base))}
    print(name, res[name], flush=True)
    del s
Path(a.out).write_text(json.dumps({"model": a.model, "set": a.set, "n": len(rows), "device": a.device, "variants": res}, indent=1))
