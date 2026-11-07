"""Submit one pass to a running ``wsp.vllm_server`` and wait for it.

    python -m wsp.vllm_client --spool S --adapter DIR|none --keys FILE --out OUT.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.vllm_client")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--adapter", required=True, help="peft directory, or 'none' for the base alone")
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args(argv)
    spool = Path(a.spool)
    jid = f"{os.getpid()}-{time.time_ns()}"
    keys = [k.strip() for k in Path(a.keys).read_text().splitlines() if k.strip()]
    job = {"adapter": None if a.adapter == "none" else a.adapter, "keys": keys, "out": a.out}
    tmp = spool / "jobs" / f"{jid}.tmp"
    tmp.write_text(json.dumps(job)); tmp.rename(spool / "jobs" / f"{jid}.json")
    t0 = time.time()
    while not (spool / "done" / jid).exists():
        if (spool / "YIELD").exists() or not (spool / "READY").exists():
            raise SystemExit("server not running")
        if time.time() - t0 > a.timeout:
            raise SystemExit("timeout")
        time.sleep(3)
    print(Path(a.out).with_suffix(".meta.json").read_text())


if __name__ == "__main__":
    main()
