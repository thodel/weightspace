"""Reading packs for the planted-defect sessions: what a programmer at budget 0 is given for one model.

    .venv/bin/python scripts/make_reading_packs.py

Per corrupted model: wsp.stats describe (the model alone), wsp.stats diff against its parent
kraken-german-xix-v1 (rows aligned by character), and the C0 reading of the practice set (CER, WER, collapsed,
top-10 confusions). Nothing about which edit was planted, and no held-out number.
"""
import json
from pathlib import Path

from wsp import stats
from wsp.kraken_subject import KrakenSubject
from wsp.plant import CORRUPTIONS, OUT, PARENT

SC = Path.home() / "atr-cache" / "wsp" / "scores"
DEST = Path("runs/planted")


def main():
    parent = KrakenSubject(PARENT)
    for cid in CORRUPTIONS:
        d = DEST / cid / "pack"
        d.mkdir(parents=True, exist_ok=True)
        s = KrakenSubject(OUT / f"{cid}.mlmodel")
        (d / "describe.md").write_text(stats.md_describe(stats.describe(s), "the subject"))
        (d / "diff_parent_to_subject.md").write_text(stats.md_diff(stats.lineage_diff(parent, s), "kraken-german-xix-v1 (parent) → the subject"))
        m = json.loads((SC / f"prac-{cid}.json").read_text())["metrics"]
        conf = "\n".join(f"| `{a or '∅'}` | `{b or '∅'}` | {n} |" for a, b, n in m["confusions"])
        (d / "c0_practice.md").write_text(
            f"# C0 on the practice set (300 lines of blb-train-300, 12,867 reference characters)\n\n"
            f"CER {m['cer']:.4f}, WER {m['wer']:.4f}, collapsed lines {m['collapsed']}, length ratio {m['length_ratio']:.3f}, "
            f"{m['char_edits']} character edits.\n\nTop-10 confusions (reference → hypothesis, ∅ = nothing):\n\n"
            f"| reference | hypothesis | count |\n|---|---|---:|\n{conf}\n")
        print(cid, "ok")


if __name__ == "__main__":
    main()
