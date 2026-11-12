#!/usr/bin/env python3
"""Stage-0 checks on the fetched TrOCR family: tokenizer identity, tensor shapes and parameter counts per component.

    python scripts/trocr_checks.py [--root ~/atr-cache/trocr] --out docs/trocr/stage0_checks.json

Reads only the safetensors headers and the small text files (never loads the weights into memory except for the
one model whose per-component counts are taken by ``TrocrSubject.describe``). Adds the sha256 of the remaining
tokenizer/config files to docs/trocr/DIGESTS.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FILES = ["vocab.json", "merges.txt", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json",
         "preprocessor_config.json", "generation_config.json", "config.json"]


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def header(p: Path) -> dict:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "atr-cache/trocr"))
    ap.add_argument("--out", default="docs/trocr/stage0_checks.json")
    ap.add_argument("--digests", default="docs/trocr/DIGESTS.json")
    ap.add_argument("--describe", default="dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2")
    ap.add_argument("--lines", default="data/practice/blb-train-300.jsonl")
    a = ap.parse_args()
    root = Path(a.root)
    dig = json.loads(Path(a.digests).read_text())
    models = {rid: root / rid.replace("/", "__") for rid in dig}
    res: dict = {"models": {}}
    for rid, d in models.items():
        e = {"files": {f: sha(d / f) for f in FILES if (d / f).exists()}}
        e["missing_files"] = [f for f in FILES if not (d / f).exists()]
        h = header(d / "model.safetensors")
        h.pop("__metadata__", None)
        e["n_tensors"] = len(h)
        e["tensor_shapes_sha256"] = hashlib.sha256(json.dumps({k: v["shape"] for k, v in sorted(h.items())}).encode()).hexdigest()
        e["dtypes"] = sorted({v["dtype"] for v in h.values()})
        e["parameters_from_header"] = sum(int(__import__("math").prod(v["shape"])) for v in h.values())
        for k in h:
            if k.endswith("embed_tokens.weight") or k.endswith("output_projection.weight"):
                e.setdefault("embedding_shapes", {})[k] = h[k]["shape"]
        dig[rid].setdefault("sha256", {}).update({f: v for f, v in e["files"].items()})
        res["models"][rid] = e
    # identity per file
    ref = "dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2"
    res["reference"] = ref
    ident = {}
    for f in FILES:
        groups: dict[str, list[str]] = {}
        for rid, e in res["models"].items():
            groups.setdefault(e["files"].get(f, "MISSING"), []).append(rid)
        ident[f] = sorted(groups.values(), key=lambda g: -len(g))
    res["file_identity_groups"] = ident
    shapes = {e["tensor_shapes_sha256"] for e in res["models"].values()}
    res["all_shapes_identical"] = len(shapes) == 1       # False when the base lacks the (unused) ViT pooler
    keysets = {}
    for rid, d in models.items():
        h = header(d / "model.safetensors"); h.pop("__metadata__", None)
        keysets[rid] = {k: tuple(v["shape"]) for k, v in h.items() if "pooler" not in k}
    res["shapes_identical_without_pooler"] = len({json.dumps(sorted(k.items())) for k in keysets.values()}) == 1
    res["pooler_in_file"] = {rid: any("pooler" in k for k in header(d / "model.safetensors")) for rid, d in models.items()}
    res["separate_output_projection_in_file"] = {rid: any("output_projection" in k for k in header(d / "model.safetensors"))
                                                 for rid, d in models.items()}
    # semantic identity of the tokenizer (vocabulary, merge list, special tokens), whatever the serialisation of the file
    sem = {}
    for rid, d in models.items():
        tj = json.loads((d / "tokenizer.json").read_text()) if (d / "tokenizer.json").exists() else None
        v = json.loads((d / "vocab.json").read_text()) if (d / "vocab.json").exists() else (tj["model"]["vocab"] if tj else None)
        m = (d / "merges.txt").read_text().split("\n") if (d / "merges.txt").exists() else None
        sem[rid] = {"vocab": v, "merges": m, "tj_vocab": tj and tj["model"]["vocab"], "tj_merges": tj and tj["model"]["merges"],
                    "added": tj and [(a["id"], a["content"], a["special"]) for a in tj["added_tokens"]]}
    r0 = sem[ref]
    def merges_of(x):
        return x["merges"] if x["merges"] is not None else [" ".join(p) if isinstance(p, list) else p for p in x["tj_merges"]]
    res["tokenizer_semantic"] = {rid: {"vocab_equal": x["vocab"] == r0["vocab"], "merges_equal": [l for l in merges_of(x) if l and not l.startswith("#version")] == [l for l in merges_of(r0) if l and not l.startswith("#version")],
                                       "tokenizer_json_vocab_equal": (x["tj_vocab"] == r0["vocab"]) if x["tj_vocab"] else None,
                                       "added_tokens_equal": (x["added"] == r0["added"]) if x["added"] else None,
                                       "vocab_json_present": x["vocab"] is not None and (models[rid] / "vocab.json").exists(),
                                       "merges_txt_present": (models[rid] / "merges.txt").exists()}
                                 for rid, x in sem.items()}
    del sem
    # behavioural: encode 300 practice lines + the special ids with each tokenizer
    from transformers import AutoTokenizer
    lines = [json.loads(l)["text"] for l in open(a.lines, encoding="utf-8")]
    enc = {}
    for rid, d in models.items():
        t = AutoTokenizer.from_pretrained(str(d))
        ids = [t(x).input_ids for x in lines]
        enc[rid] = {"vocab_len": len(t), "special": {"bos": t.bos_token_id, "pad": t.pad_token_id, "eos": t.eos_token_id},
                    "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    res["encoding"] = enc
    res["encoding_identical"] = len({e["ids_sha256"] for e in enc.values()}) == 1
    # parameter counts per component (counted by TrocrSubject)
    from wsp.trocr_subject import TrocrSubject
    s = TrocrSubject(models[a.describe])
    dsc = s.describe()
    res["describe"] = {k: dsc[k] for k in ("parameters", "encoder_parameters", "decoder_parameters", "bridge_and_other", "by_component",
                                           "tensors", "state_dict_keys", "forward_parameters", "unused_by_forward", "aliases", "unused_tensors", "decoder_tie_word_embeddings", "other_tensors", "encoder_layers", "decoder_layers", "vocab_size")}
    res["describe"]["model"] = a.describe
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    Path(a.digests).write_text(json.dumps(dig, indent=1, sort_keys=True))
    print(json.dumps({k: res[k] for k in ("all_shapes_identical", "encoding_identical")}))
    for f, g in ident.items():
        print(f, [[x.split("/")[1][-24:] for x in grp] for grp in g])
    print(json.dumps(res["describe"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
