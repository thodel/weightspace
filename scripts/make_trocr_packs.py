"""Reading packs for the TrOCR programmer arms (docs/TROCR_STAGE3_PREREG.md, section 3): what a programmer is given for one planted model.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/make_trocr_packs.py [planted ids...]

Per core model, under an opaque label (runs/trocr-planted/sNN/, map in runs/trocr-planted/map.json, written here but not shown to a programmer):
  pack/describe.md              the model alone: per tensor shape, L2 norm, std, zero share, non-finite count; per-layer totals
  pack/diff_parent_to_subject.md  the model against the parent donor (trocr-kurrent-XVI-XVII): per tensor relative distance, norm ratio, cosine;
                                  the token rows of the tied embedding that moved most; attention heads (64-column slices) that moved most
  pack/c0_practice.md           the model's reading of the 300 practice lines (CER, WER, collapsed, top confusions), from the C0 pass of the arm-(c) session
  pack_nodonor/                 describe.md and c0_practice.md only (arm (a) without the donor)
Nothing says which defect was planted, and no held-out number is in a pack. Non-ASCII and white-space characters are always printed with their
code points (the kraken pack printed the no-break space like a plain space, docs/CONTROLS_REPORT.md section 11.2; checked in ``check_visible``).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

from wsp.trocr_family import pair_stats
from wsp.trocr_plant import OUT, PARENT
from wsp.trocr_subject import TrocrSubject

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "runs/trocr-planted"
CORE = ["p01-zero-rows-k6", "p03-cross-o5-zero", "p05-zero-enc6-attn", "p06-enc3-ffn-x0.5", "p07-noise-dec8-60", "p08-shift-embnorm",
        "p10-noise-all-20", "p12-enc-ffn-all-x0.7"]
LABELS = ["s03", "s07", "s01", "s05", "s08", "s02", "s06", "s04"]     # fixed shuffle, so that the folder order does not follow the defect list
HEADS, HD = 16, 64


def vis(s: str) -> str:
    """A string with every character that is not printable ASCII (and every space) shown by code point."""
    if s and all(32 < ord(c) < 127 for c in s):
        return f"`{s}`"
    return "`" + "".join(c if 32 < ord(c) < 127 else f"<U+{ord(c):04X}>" for c in s) + "`"


def check_visible():
    assert vis("a b") == "`a<U+00A0>b`" and vis(" ") == "`<U+0020>`" and vis("Ġe") == "`<U+0120>e`"


def tstats(T):
    rows = []
    for n in sorted(T):
        a = T[n].astype(np.float64)
        rows.append((n, a.shape, float(np.linalg.norm(a)), float(a.std()), float(np.mean(a == 0)), int((~np.isfinite(a)).sum())))
    return rows


def describe_md(sub, T):
    rows = tstats(T)
    L = ["# The model alone", "", f"{len(T)} editable tensors, {sum(a.size for a in T.values())} parameters. Names are canonical (see the brief). "
         "The decoder output projection is the same tensor as `dec.embed.tokens` (tied); there is no output bias.", "",
         "| tensor | shape | L2 norm | std | share of zeros | non-finite |", "|---|---|---:|---:|---:|---:|"]
    for n, sh, nm, sd, z, nf in rows:
        L.append(f"| `{n}` | {'×'.join(map(str, sh))} | {nm:.4g} | {sd:.4g} | {z:.4f} | {nf} |")
    L += ["", "## Per layer (L2 norm of all weight matrices and vectors of the layer)", "", "| layer | norm |", "|---|---:|"]
    byl = {}
    for n, _, nm, *_ in rows:
        p = n.split(".")
        byl.setdefault(".".join(p[:2]) if p[0] in ("enc", "dec") and p[1].isdigit() else p[0] + "." + p[1], []).append(nm * nm)
    for k in sorted(byl):
        L.append(f"| {k} | {np.sqrt(sum(byl[k])):.4g} |")
    return "\n".join(L) + "\n"


def diff_md(T, P, tok):
    L = ["# The model against its parent donor `trocr-kurrent-XVI-XVII`", "",
         "Relative distance = ‖a−b‖ / mean(‖a‖,‖b‖); norm ratio = ‖subject‖/‖parent‖; cosine of the flattened tensors. "
         "A healthy child of this parent (the recorded step to `from-kurrent-v2`) moves weight matrices by 0.07 to 0.24, "
         "layer-norm vectors by 0.007 to 0.13 and the tied embedding by about 0.5 (docs/trocr/readings.json); "
         "this is the only reference for 'normal' you are given.", "",
         "| tensor | rel. distance | norm ratio | cosine | identical share |", "|---|---:|---:|---:|---:|"]
    for n in sorted(T):
        st = pair_stats(T[n], P[n])
        nr = float(np.sqrt(st["na2"] / st["nb2"])) if st["nb2"] > 0 else float("nan")
        L.append(f"| `{n}` | {st['dist']:.4f} | {nr:.4f} | {'' if st['cos'] is None else format(st['cos'], '.4f')} | {st['identical']:.4f} |")
    E, EP = T["dec.embed.tokens"].astype(np.float64), P["dec.embed.tokens"].astype(np.float64)
    ne, nep = np.linalg.norm(E, axis=1), np.linalg.norm(EP, axis=1)
    cos = (E * EP).sum(1) / np.maximum(ne * nep, 1e-12)
    dist = np.linalg.norm(E - EP, axis=1) / np.maximum((ne + nep) / 2, 1e-12)
    L += ["", "## Token rows of the tied embedding (`dec.embed.tokens`, one row per token id): the 25 with the lowest cosine to the parent's row", "",
          "| id | token | cosine | row norm ratio | rel. distance |", "|---:|---|---:|---:|---:|"]
    for i in np.argsort(cos)[:25]:
        L.append(f"| {i} | {vis(tok.convert_ids_to_tokens(int(i)) or '')} | {cos[i]:.4f} | {ne[i] / max(nep[i], 1e-12):.4f} | {dist[i]:.4f} |")
    L += ["", f"Rows with cosine below 0.5: {int((cos < 0.5).sum())}; rows with norm below 0.5 of the parent's: {int((ne < 0.5 * nep).sum())}; "
          f"rows that are exactly zero: {int((ne == 0).sum())}.", "",
          "## Attention heads (64-column slices of `*.o.weight`: 12 heads in the encoder, 16 in the decoder): heads whose relative distance is over 3 times the median of their tensor", "",
          "| tensor | head | rel. distance | median of tensor |", "|---|---:|---:|---:|"]
    n_listed = 0
    for n in sorted(T):
        if not n.endswith(".o.weight"):
            continue
        a, b = T[n].astype(np.float64), P[n].astype(np.float64)
        nh = a.shape[1] // HD                               # 12 heads in the encoder (768), 16 in the decoder (1024)
        d = np.array([np.linalg.norm(a[:, h * HD:(h + 1) * HD] - b[:, h * HD:(h + 1) * HD]) /
                      max((np.linalg.norm(a[:, h * HD:(h + 1) * HD]) + np.linalg.norm(b[:, h * HD:(h + 1) * HD])) / 2, 1e-12) for h in range(nh)])
        med = float(np.median(d))
        for h in np.where(d > 3 * max(med, 1e-9))[0]:
            L.append(f"| `{n}` | {h} | {d[h]:.4f} | {med:.4f} |")
            n_listed += 1
    if not n_listed:
        L.append("| (none) | | | |")
    return "\n".join(L) + "\n"


def c0_md(pid):
    r = sorted((REPO / "runs").glob(f"*-trocr-planted-{pid}-c-b10-1"))[-1]
    m = json.loads((r / "summary.json").read_text())["baseline"]
    conf = "\n".join(f"| {vis(a) if a else '∅'} | {vis(b) if b else '∅'} | {n} |" for a, b, n in m["confusions"])
    return (f"# C0 on the practice set (300 lines of the training pages of the BLB hand, {m['ref_chars']} reference characters)\n\n"
            f"CER {m['cer']:.4f}, WER {m['wer']:.4f}, collapsed lines {m['collapsed']}, length ratio {m['length_ratio']:.3f}, {m['char_edits']} character edits.\n\n"
            f"Top-10 confusions (reference → hypothesis, ∅ = nothing; characters shown with code points when not plain ASCII):\n\n"
            f"| reference | hypothesis | count |\n|---|---|---:|\n{conf}\n")


def main(ids):
    check_visible()
    ids = ids or CORE
    parent = TrocrSubject(PARENT, device="cpu", check_card=False)
    P = parent.tensors()
    mp = {}
    for pid in ids:
        lab = LABELS[CORE.index(pid)]
        sub = TrocrSubject(OUT / pid, device="cpu", check_card=False)
        T = sub.tensors()
        for d in ("pack", "pack_nodonor"):
            (DEST / lab / d).mkdir(parents=True, exist_ok=True)
        desc = describe_md(sub, T)
        c0 = c0_md(pid)
        (DEST / lab / "pack/describe.md").write_text(desc)
        (DEST / lab / "pack/c0_practice.md").write_text(c0)
        (DEST / lab / "pack/diff_parent_to_subject.md").write_text(diff_md(T, P, sub.tokenizer))
        (DEST / lab / "pack_nodonor/describe.md").write_text(desc)
        (DEST / lab / "pack_nodonor/c0_practice.md").write_text(c0)
        mp[lab] = pid
        print(pid, "->", lab, flush=True)
    f = DEST / "map.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **mp}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
