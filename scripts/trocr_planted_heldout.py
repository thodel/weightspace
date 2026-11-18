"""Held-out scoring (blb-eval-hand, once per digest, cached by wsp.run's rule) of the planted TrOCR models and of the baseline finals
(docs/TROCR_STAGE3_PREREG.md, baseline results). Targets: the unplanted subject, the core 8 corrupted models and the corrupted models of
any other defect with a changed final, every changed final of arms c and d. Writes docs/trocr/heldout_baselines.json.
    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_planted_heldout.py
"""
import json, re, sys, time
from pathlib import Path
from wsp import run as R
from wsp.metrics import score, per_line_edits
from wsp.data import load, sha256_file
REPO = Path(__file__).resolve().parents[1]
PL = Path.home() / "atr-cache" / "wsp" / "planted-trocr"
SET = "blb-eval-hand"
OUT = REPO / "docs" / "trocr" / "heldout_baselines.json"
CORE = ["p01-zero-rows-k6", "p03-cross-o5-zero", "p05-zero-enc6-attn", "p06-enc3-ffn-x0.5", "p07-noise-dec8-60", "p08-shift-embnorm", "p10-noise-all-20", "p12-enc-ffn-all-x0.7"]

def targets():
    t, seen = [("unplanted subject (p14-null)", PL / "p14-null")], set()
    fin = []
    for r in sorted((REPO / "runs").glob("*-trocr-planted-*-b10-1")):
        pid, arm = re.search(r"planted-(.+)-([cd])-b10", r.name).groups()
        s = json.loads((r / "summary.json").read_text())
        if s["kept"] > 0 and (r / "final.trocr").is_dir():
            fin.append((pid, f"{pid} arm {arm} final", r / "final.trocr"))
    need = list(CORE) + [p for p, _, _ in fin if p not in CORE]
    for p in need:
        if p not in seen:
            seen.add(p); t.append((f"{p} corrupted", PL / p))
    t += [(l, d) for _, l, d in fin]
    return t

def score_model(path):
    items = load(R.HELDOUT / f"{SET}.jsonl"); set_sha = sha256_file(R.HELDOUT / f"{SET}.jsonl")
    s = R._subject(path, meta={"device": "cuda:0", "batch": 8})
    cache = R.SCORES / f"heldout-{SET}-{set_sha[:12]}-{s.digest()[:16]}.json"
    refs = [x["text"] for x in items]
    if cache.exists():
        c = json.loads(cache.read_text()); hyps, cached = c["hyps"], True
    else:
        hyps, cached = [p.text for p in s.forward(items)], False
        R.SCORES.mkdir(parents=True, exist_ok=True)
        m0 = score(hyps, refs)
        cache.write_text(json.dumps({"digest": s.digest(), "set": SET, "set_sha256": set_sha, "keys": [x["key"] for x in items], "hyps": hyps, "metrics": m0.as_dict()}, ensure_ascii=False))
    return {"digest": s.digest(), "cached": cached, "metrics": score(hyps, refs).as_dict(), "per_line": per_line_edits(hyps, refs)}

if __name__ == "__main__":
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for label, path in targets():
        if label in res: continue
        t = time.time(); res[label] = {"path": str(path), **score_model(path), "wall_s": round(time.time() - t, 1)}
        OUT.write_text(json.dumps(res, ensure_ascii=False))
        print("scored", label, round(res[label]["metrics"]["cer"], 4), res[label]["wall_s"], "s", flush=True)
    print("heldout finished", time.strftime("%FT%T%z"))
