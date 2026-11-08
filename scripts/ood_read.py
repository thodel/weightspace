"""Read line sets with unedited models, in parallel CPU processes (docs/MODEL_FAMILY.md, 08.11.2026).

    CUDA_VISIBLE_DEVICES="" python scripts/ood_read.py --models <a.mlmodel> ... --sets data/practice/<set>.jsonl ... \\
        --out docs/family/ood-screen.json [--threads 3 --procs 12]

Each (model, set) pair is one task: the per-line readings go to
``docs/family/readings/ood/<set>/<model>.jsonl`` (git-ignored), the corpus numbers to ``--out``
(merged into the file if it exists). Used to find practice material a model reads at 15-30 % CER;
never pointed at a held-out list.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(REPO))          # this checkout's wsp, not the venv's editable install
READINGS = REPO / "docs" / "family" / "readings" / "ood"


def _task(args):
    model, lst, threads = args
    import torch
    torch.set_num_threads(threads)
    import sys
    sys.path.insert(0, str(REPO))
    from wsp.data import load, sha256_file
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import score
    if "heldout" in Path(lst).parts:
        raise SystemExit(f"{lst}: a held-out list; this script reads practice lists only")
    items = load(lst)
    t = time.time()
    s = KrakenSubject(model)
    preds = s.forward(items)
    hyps = [p.text for p in preds]; refs = [x["text"] for x in items]
    m = score(hyps, refs)
    out = READINGS / Path(lst).stem / f"{Path(model).stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"key": p.key, "hyp": h, "ref": r}, ensure_ascii=False) + "\n"
                           for p, h, r in zip(preds, hyps, refs)), encoding="utf-8")
    return {"model": Path(model).stem, "set": Path(lst).stem, "set_sha256": sha256_file(Path(lst)), "lines": m.lines,
            "cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "length_ratio": m.length_ratio,
            "digest": s.digest(), "wall_s": round(time.time() - t, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {"rows": []}
    tasks = [(m, s, a.threads) for s in a.sets for m in a.models]
    with ProcessPoolExecutor(a.procs) as ex:
        for r in ex.map(_task, tasks):
            print(f"{r['set']:28s} {r['model']:36s} CER {100*r['cer']:6.2f} %  ({r['lines']} lines, {r['wall_s']} s)", flush=True)
            res["rows"] = [x for x in res["rows"] if (x["model"], x["set"]) != (r["model"], r["set"])] + [r]
            res["when"] = time.strftime("%Y-%m-%d %H:%M")
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
