"""Reading packs for the validation-practice sessions (item 3): describe, diffs, C0 on the practice set, donors' CER on it."""
import json
from pathlib import Path

from wsp import stats
from wsp.kraken_subject import KrakenSubject
from wsp.data import load

M = Path.home() / "Repo/weightspace/models/kraken"
SC = Path.home() / "atr-cache/wsp/scores"
DEST = Path("runs/validation")


def c0(path, title, n_ref):
    m = json.loads(path.read_text())["metrics"]
    conf = "\n".join(f"| `{a or '∅'}` | `{b or '∅'}` | {n} |" for a, b, n in m["confusions"])
    return (f"# C0 on the practice set ({title}, {n_ref} lines)\n\nCER {m['cer']:.4f}, WER {m['wer']:.4f}, collapsed lines {m['collapsed']}, "
            f"length ratio {m['length_ratio']:.3f}, {m['char_edits']} character edits.\n\nTop-10 confusions (reference → hypothesis, ∅ = nothing):\n\n"
            f"| reference | hypothesis | count |\n|---|---|---:|\n{conf}\n")


def donor_cer(model, lst, items, mask=None):
    from wsp.metrics import score
    r = json.loads((SC / f"val-{model}-{lst}.json").read_text())
    idx = [i for i in range(len(items)) if mask is None or mask[i]]
    m = score([r["hyps"][i] for i in idx], [items[i]["text"] for i in idx])
    return m.cer, len(idx)


def blb():
    d = DEST / "xix-blb-v1" / "pack"; d.mkdir(parents=True, exist_ok=True)
    S = KrakenSubject(M / "kraken-german-xix-blb-v1.mlmodel"); P = KrakenSubject(M / "kraken-german-xix-v1.mlmodel")
    (d / "describe.md").write_text(stats.md_describe(stats.describe(S), "the subject"))
    (d / "diff_parent_to_subject.md").write_text(stats.md_diff(stats.lineage_diff(P, S), "kraken-german-xix-v1 (parent) → the subject"))
    items = load("data/practice/blb-val-300.jsonl")
    pc, n = donor_cer("kraken-german-xix-v1", "blbval", items)
    (d / "c0_practice.md").write_text(c0(SC / "val-kraken-german-xix-blb-v1-blbval.json", "blb-val-300: handwritten lines from 7 objects the model was never trained on", 300)
        + f"\nDonor available: `kraken-german-xix-v1` (the parent; never trained on BLB data), CER {pc:.4f} on the same lines. `seen_by` is empty for every line.\n")


def bern():
    d = DEST / "bern-combined-v1" / "pack"; d.mkdir(parents=True, exist_ok=True)
    S = KrakenSubject(M / "kraken-bern-combined-v1.mlmodel"); V2 = KrakenSubject(M / "kraken-bern-hertwig-v2.mlmodel"); SR = KrakenSubject(M / "kraken-bern-schlussredaktion-v1.mlmodel")
    (d / "describe.md").write_text(stats.md_describe(stats.describe(S), "the subject (combined-v1)"))
    (d / "diff_hertwig-v2_to_subject.md").write_text(stats.md_diff(stats.lineage_diff(V2, S), "kraken-bern-hertwig-v2 (parent) → the subject"))
    (d / "diff_hertwig-v2_to_schlussredaktion-v1.md").write_text(stats.md_diff(stats.lineage_diff(V2, SR), "kraken-bern-hertwig-v2 → kraken-bern-schlussredaktion-v1 (sibling)"))
    items = load("data/practice/bern-val.jsonl")
    out = [c0(SC / "val-kraken-bern-combined-v1-bernval.json", "bern-val: the 17 pages the trainer reserved for validation of the subject; 6 Hertwig pages, 11 Schlussredaktion pages", len(items)), "## Donors\n"]
    for mid in ("kraken-bern-hertwig-v2", "kraken-bern-schlussredaktion-v1"):
        mask = [mid not in it["seen_by"] for it in items]
        a, n = donor_cer(mid, "bernval", items); b, n2 = donor_cer(mid, "bernval", items, mask)
        out.append(f"- `{mid}`: CER {a:.4f} on all {n} lines; trained on {n - n2} of them (`seen_by`), CER {b:.4f} on the {n2} lines it was not trained on. "
                   f"For this donor use `rule_mask=s.unseen_by('{mid}')`.")
    seen = sum(1 for it in items if it["seen_by"])
    out.append(f"\nPractice lines: {len(items)}; Hertwig {sum(1 for i in items if i['doc']=='13922499')}, Schlussredaktion {sum(1 for i in items if i['doc']=='13922579')}. "
               "Donor paths: `/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel`, `.../kraken-bern-schlussredaktion-v1.mlmodel`. "
               "Note the subject was initialised from hertwig-v2, which trained on every Hertwig page except its own 5 validation pages.")
    (d / "c0_practice.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    blb(); bern()
