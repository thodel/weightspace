"""Reading packs for the stage-4 programmer sessions on subject A (docs/TROCR_STAGE4_PREREG.md section 5): runs/trocr-stage4/A/{pack,pack_nodonor}.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/make_trocr_stage4_packs.py

pack/           describe.md (the model alone), c0_practice.md (its reading of the 1,194 practice lines), diff_magnitudes.md (against the donor: MAGNITUDES only,
                no cosine and no direction reading, R16)
pack_nodonor/   describe.md and c0_practice.md only
No held-out number is in a pack."""
import json
from pathlib import Path

import numpy as np

from scripts.make_trocr_packs import describe_md, vis, check_visible, HD
from wsp.trocr_subject import TrocrSubject, component

REPO = Path(__file__).resolve().parent.parent
AC = Path.home() / "atr-cache/trocr"
SUBJ, DONOR = AC / "dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2", AC / "dh-unibe__trocr-kurrent-XVI-XVII"
DEST = REPO / "runs/trocr-stage4/A"
C0RUN = REPO / "runs/20261119-trocr-stage4-noop-A-nodonor-b1-3"


def c0_md():
    m = json.loads((C0RUN / "summary.json").read_text())["baseline"]
    conf = "\n".join(f"| {vis(a) if a else '∅'} | {vis(b) if b else '∅'} | {n} |" for a, b, n in m["confusions"])
    return (f"# C0 on the practice set ({m['lines']} lines of the training pages of the BLB hand, {m['ref_chars']} reference characters)\n\n"
            f"CER {m['cer']:.4f}, WER {m['wer']:.4f}, collapsed lines {m['collapsed']}, length ratio {m['length_ratio']:.3f}, {m['char_edits']} character edits.\n\n"
            f"Top-10 confusions (reference → hypothesis, ∅ = nothing; characters shown with code points when not plain ASCII):\n\n"
            f"| reference | hypothesis | count |\n|---|---|---:|\n{conf}\n")


def diff_md(T, P, tok):
    L = ["# The model against its donor `trocr-kurrent-XVI-XVII`: magnitudes only", "",
         "Relative distance = ‖a−b‖ / mean(‖a‖,‖b‖); norm ratio = ‖subject‖/‖donor‖. No direction reading is given (directions of moves do not reproduce across "
         "retrained models). The recorded step of this family moves weight matrices by about 0.07 to 0.24, layer-norm vectors by 0.007 to 0.13 and the tied "
         "embedding by about 0.5.", "",
         "## Where the move is (share of the squared distance ‖a−b‖², by component)", "", "| component | share | tensors |", "|---|---:|---:|"]
    sq, cnt = {}, {}
    for n in T:
        d = float(np.sum((T[n].astype(np.float64) - P[n].astype(np.float64)) ** 2))
        c = component(n)
        sq[c] = sq.get(c, 0) + d
        cnt[c] = cnt.get(c, 0) + 1
    tot = sum(sq.values())
    for c in sorted(sq, key=lambda c: -sq[c]):
        L.append(f"| {c} | {sq[c] / tot:.4f} | {cnt[c]} |")
    L += ["", "## Per tensor", "", "| tensor | rel. distance | norm ratio |", "|---|---:|---:|"]
    for n in sorted(T):
        a, b = T[n].astype(np.float64), P[n].astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        L.append(f"| `{n}` | {np.linalg.norm(a - b) / max((na + nb) / 2, 1e-12):.4f} | {na / max(nb, 1e-12):.4f} |")
    E, EP = T["dec.embed.tokens"].astype(np.float64), P["dec.embed.tokens"].astype(np.float64)
    ne, nep = np.linalg.norm(E, axis=1), np.linalg.norm(EP, axis=1)
    dist = np.linalg.norm(E - EP, axis=1) / np.maximum((ne + nep) / 2, 1e-12)
    order = np.argsort(dist)
    L += ["", "## Token rows of the tied embedding by movement decile (rel. distance of the row to the donor's row)", "",
          "| decile (0 = least moved) | rows | rel. distance range | mean row norm ratio | 5 most moved tokens of the decile |", "|---:|---:|---|---:|---|"]
    for k, idx in enumerate(np.array_split(order, 10)):
        top = idx[np.argsort(-dist[idx])][:5]
        L.append(f"| {k} | {len(idx)} | {dist[idx].min():.3f} – {dist[idx].max():.3f} | {np.mean(ne[idx] / np.maximum(nep[idx], 1e-12)):.4f} | "
                 + " ".join(vis(tok.convert_ids_to_tokens(int(i)) or '') for i in top) + " |")
    L += ["", "## Attention heads (64-column slices of `*.o.weight`): the 20 with the largest relative distance", "", "| tensor | head | rel. distance |", "|---|---:|---:|"]
    rows = []
    for n in sorted(T):
        if not n.endswith(".o.weight"):
            continue
        a, b = T[n].astype(np.float64), P[n].astype(np.float64)
        for h in range(a.shape[1] // HD):
            x, y = a[:, h * HD:(h + 1) * HD], b[:, h * HD:(h + 1) * HD]
            rows.append((np.linalg.norm(x - y) / max((np.linalg.norm(x) + np.linalg.norm(y)) / 2, 1e-12), n, h))
    for d, n, h in sorted(rows, reverse=True)[:20]:
        L.append(f"| `{n}` | {h} | {d:.4f} |")
    return "\n".join(L) + "\n"


def main():
    check_visible()
    sub, don = TrocrSubject(SUBJ, device="cpu", check_card=False), TrocrSubject(DONOR, device="cpu", check_card=False)
    T, P = sub.tensors(), don.tensors()
    for d in ("pack", "pack_nodonor"):
        (DEST / d).mkdir(parents=True, exist_ok=True)
        (DEST / d / "describe.md").write_text(describe_md(sub, T))
        (DEST / d / "c0_practice.md").write_text(c0_md())
    (DEST / "pack/diff_magnitudes.md").write_text(diff_md(T, P, sub.tokenizer))
    for f in DEST.rglob("*.md"):
        assert "cosine" not in f.read_text().lower() or f.name == "describe.md" and False, f
    print("packs in", DEST)


if __name__ == "__main__":
    main()
