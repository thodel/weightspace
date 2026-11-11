#!/usr/bin/env python3
"""C0 for a TrOCR subject: load, (optionally save and reload), read a line set by generation, score by wsp.metrics.

    python scripts/trocr_c0.py MODEL_DIR SET.jsonl --device cpu|cuda:0 --out OUT_DIR [--repeat] [--limit N] [--batch 8]
                               [--teacher-forced] [--roundtrip]

Writes OUT_DIR/<set>.<model>.<device>.json (metrics, time, settings, versions, prediction hash) and the predictions
(.pred.jsonl). ``--repeat`` runs the pass a second time in the same process and states whether the predictions are
byte-identical. Card 0 is checked by ``wsp.card.ready`` (12 GB free and no trainer job waiting) before a cuda pass.
Never card 1: CUDA_VISIBLE_DEVICES is pinned to 0 here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cet() -> str:
    from datetime import datetime, timezone, timedelta
    # CET label as the project reports times (fixed UTC+1 in November)
    return (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S CET")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("set")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--teacher-forced", action="store_true")
    ap.add_argument("--roundtrip", action="store_true", help="save to OUT/roundtrip and read from there")
    ap.add_argument("--no-card-check", action="store_true")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if not a.no_card_check:
            from wsp import card
            ok, why = card.ready()
            print("card:", why, flush=True)
            if not ok:
                return 3
    import torch
    import transformers
    from wsp import data, metrics
    from wsp.trocr_subject import TrocrSubject

    if a.threads:
        torch.set_num_threads(a.threads)
    rows = data.load(a.set)
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time()
    sub = TrocrSubject(a.model, device=a.device, batch=a.batch)
    load_s = time.time() - t0
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = {}
    if a.roundtrip:
        d0 = sub.digest()
        saved = sub.save(out_dir / "roundtrip")
        sub2 = TrocrSubject(saved, device=a.device, batch=a.batch)
        rt = {"digest_before": d0, "digest_after": sub2.digest(), "equal": d0 == sub2.digest()}
        sub = sub2
    name = Path(a.model).name.replace("dh-unibe__", "")
    stem = f"{Path(a.set).stem}.{name}.{a.device.replace(':', '')}"
    passes = []
    for i in range(2 if a.repeat else 1):
        if a.device.startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = cet(); t1 = time.time()
        preds = sub.forward(rows)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t1
        blob = "".join(json.dumps({"key": p.key, "text": p.text, "extra": p.extra}, ensure_ascii=False, sort_keys=True) + "\n"
                       for p in preds)
        passes.append({"started": start, "seconds": round(dt, 1), "sec_per_line": round(dt / len(rows), 3),
                       "pred_sha256": hashlib.sha256(blob.encode()).hexdigest(),
                       "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20) if a.device.startswith("cuda") else None})
        if i == 0:
            (out_dir / f"{stem}.pred.jsonl").write_text(blob, encoding="utf-8")
            first = preds
        print(f"pass {i + 1}: {len(rows)} lines in {dt:.1f} s ({dt / len(rows):.2f} s/line)", flush=True)
    m = metrics.score([p.text for p in first], [r["text"] for r in rows], blocks=False)
    tok = [p.extra["tokens"] for p in first]
    rep = {"model": a.model, "set": a.set, "set_sha256": hashlib.sha256(Path(a.set).read_bytes()).hexdigest(),
           "lines": len(rows), "device": a.device, "batch": a.batch, "load_seconds": round(load_s, 1),
           "digest": sub.digest() if not rt else rt["digest_after"],
           "cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "length_ratio": m.length_ratio,
           "char_edits": m.char_edits, "ref_chars": m.ref_chars,
           "tokens_total": sum(tok), "tokens_max": max(tok), "at_cap": sum(bool(p.extra["at_cap"]) for p in first),
           "top_confusions": [list(c) for c in m.confusions],
           "generation": sub.generation, "passes": passes,
           "byte_identical_repeat": (passes[0]["pred_sha256"] == passes[1]["pred_sha256"]) if a.repeat else None,
           "roundtrip": rt or None,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "threads": torch.get_num_threads(), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    if a.teacher_forced:
        tf = sub.teacher_forced(rows)
        allnll = [x for r in tf for x in r["nll"]]
        allm = [x for r in tf for x in r["margin"]]
        rep["teacher_forced"] = {"mean_token_nll": sum(allnll) / len(allnll), "tokens": len(allnll),
                                 "share_margin_negative": sum(x <= 0 for x in allm) / len(allm)}
    (out_dir / f"{stem}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("cer", "wer", "collapsed", "length_ratio", "byte_identical_repeat")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
