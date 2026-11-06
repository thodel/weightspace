"""C0 of the protocol: load, (save,) score — the un-budgeted baseline numbers per model and practice set.

    python -m wsp.baselines --model models/kraken/<id>.mlmodel [...] --practice data/practice/<set>.jsonl --out docs/baselines/<set>.json

Writes one JSON per practice set with, per model: metrics (CER, WER, collapsed,
length ratio, top-10 confusions), blank share, the tensor digest, the file
digest, the practice-set sha256, the wall time, and the per-line predictions
under ``docs/baselines/readings/<set>/<model>.jsonl`` (not budgeted, not a
session). ``render`` turns the JSONs into docs/BASELINES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wsp.data import load, sha256_file
from wsp.metrics import score, per_line_edits


def blank_share(subject, practice: list[dict]) -> dict:
    """Mean probability mass on the blank, and the share of frames where it wins."""
    tot = win = frames = 0.0
    for probs in subject.logits(practice):
        p = np.asarray(probs)
        tot += float(p[:, 0].sum()); win += float((p.argmax(1) == 0).sum()); frames += p.shape[0]
    return {"mean_blank_prob": tot / frames if frames else float("nan"),
            "blank_argmax_share": win / frames if frames else float("nan"), "frames": int(frames)}


def run(model: Path, practice_path: Path, readings_dir: Path, device: str = "cpu") -> dict:
    from wsp.kraken_subject import KrakenSubject
    practice = load(practice_path)
    t = time.time()
    s = KrakenSubject(model, device=device)
    preds = s.forward(practice)
    wall = time.time() - t
    hyps = [p.text for p in preds]; refs = [x["text"] for x in practice]
    m = score(hyps, refs)
    readings_dir.mkdir(parents=True, exist_ok=True)
    (readings_dir / f"{model.stem}.jsonl").write_text(
        "".join(json.dumps({"key": p.key, "hyp": p.text, "ref": r}, ensure_ascii=False) + "\n"
                for p, r in zip(preds, refs)), encoding="utf-8")
    d = s.describe()
    return {"model": model.stem, "file_sha256": sha256_file(model), "tensor_digest": s.digest(),
            "parameters": d["parameters"], "codec_size": d["codec_size"], "input_height": d["input_height"],
            "practice": practice_path.stem, "practice_sha256": sha256_file(practice_path),
            "lines": len(practice), "metrics": m.as_dict(), "blank": blank_share(s, practice),
            "per_line": per_line_edits(hyps, refs), "wall_s": round(wall, 1), "device": device}


def render(jsons: list[Path], out: Path) -> None:
    lines = ["# Baselines — C0, practice sets (not results)", "",
             "Practice numbers on the *train-side* practice sets of docs/BRIEF_PHASE0.md §4, "
             "scored once per model outside any session (PROTOCOL §4, C0). They are **not** "
             "held-out results and must not be quoted as such; they fix the starting point every "
             "edit is measured against. CER/WER corpus-level, NFD, whitespace flattened "
             "(`wsp.metrics`). Blank share: mean softmax mass on the CTC blank over all frames. "
             "CPU, kraken 7.0.2, 06.11.2026.", ""]
    for j in sorted(jsons):
        d = json.loads(j.read_text())
        rows = d["models"]
        lines += [f"## {d['practice']} — {rows[0]['lines']} lines, list sha256 `{rows[0]['practice_sha256']}`", "",
                  "| model | CER | WER | collapsed | length ratio | blank share | params | s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            m = r["metrics"]
            lines.append(f"| {r['model']} | {m['cer']:.4f} | {m['wer']:.4f} | {m['collapsed']} | "
                         f"{m['length_ratio']:.3f} | {r['blank']['mean_blank_prob']:.3f} | {r['parameters']:,} | {r['wall_s']} |")
        lines.append("")
        for r in rows:
            conf = ", ".join(f"`{a or '∅'}`→`{b or '∅'}` {n}" for a, b, n in r["metrics"]["confusions"])
            lines.append(f"- **{r['model']}** top confusions (ref→hyp): {conf}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.baselines")
    ap.add_argument("--model", nargs="*", default=[])
    ap.add_argument("--practice")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--render", nargs="*", help="JSONs to render into docs/BASELINES.md")
    a = ap.parse_args(argv)
    if a.render is not None:
        render([Path(p) for p in a.render], Path("docs/BASELINES.md")); print("docs/BASELINES.md"); return
    practice = Path(a.practice); out = Path(a.out)
    readings = out.parent / "readings" / practice.stem
    results = []
    for mp in a.model:
        r = run(Path(mp), practice, readings, a.device)
        results.append(r)
        m = r["metrics"]
        print(f"{r['model']:40s} CER {m['cer']:.4f}  WER {m['wer']:.4f}  collapsed {m['collapsed']:3d}  "
              f"ratio {m['length_ratio']:.3f}  blank {r['blank']['mean_blank_prob']:.3f}  {r['wall_s']}s")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"practice": practice.stem, "models": results}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
