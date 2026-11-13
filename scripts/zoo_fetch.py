"""Download the gateway's third-party kraken models from Zenodo into models/zoo/ (git-ignored).

    python scripts/zoo_fetch.py docs/zoo/gateway_zenodo.json models/zoo docs/zoo/fetch.json

Read-only on the network side (public Zenodo records through htrmopo); writes only under models/zoo and the
manifest. A failure is recorded and the loop goes on, except an HTTP 401/403/429, which stops the run.
"""
import hashlib, json, sys, traceback
from pathlib import Path
from htrmopo import get_model

src, dest, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = json.loads(src.read_text())
res = json.loads(manifest.read_text()) if manifest.exists() else {}
for m in rows:
    if m["engine"] != "kraken" or (m["id"] in res and res[m["id"]].get("files")):
        continue
    d = dest / m["id"]
    try:
        p = Path(get_model(m["zenodo_id"], path=d))
        files = []
        for f in sorted(p.rglob("*")):
            if f.is_file():
                files.append({"name": str(f.relative_to(p)), "bytes": f.stat().st_size,
                              "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
        res[m["id"]] = {"doi": m["zenodo_id"], "files": files}
        print("ok", m["id"], [(x["name"], x["bytes"]) for x in files], flush=True)
    except Exception as e:  # noqa: BLE001
        res[m["id"]] = {"doi": m["zenodo_id"], "files": [], "error": f"{type(e).__name__}: {e}"}
        print("FAIL", m["id"], res[m["id"]]["error"][:300], flush=True)
        if any(c in str(e) for c in ("401", "403", "429")):
            manifest.write_text(json.dumps(res, indent=1)); sys.exit(3)
    manifest.write_text(json.dumps(res, indent=1))
