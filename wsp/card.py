"""Card 0 before a pass: enough memory, and no trainer job waiting for it (#396).

The trainer's ``GET /health`` carries ``waiting`` (``null`` or
``{"job_id", "engine", "needs_mib"}``). A job waiting while card 0 is ours is
a job we would hold back, so the pass does not start. Card 1 is never used.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request

HEALTH = "http://127.0.0.1:8204/health"


def free_mib(card: int = 0) -> int:
    out = subprocess.run(["nvidia-smi", f"--id={card}", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def waiting(timeout: float = 90.0):
    with urllib.request.urlopen(HEALTH, timeout=timeout) as r:   # /health takes ~40 s on asteraix
        return json.load(r).get("waiting")


def ready(need_mib: int = 12000) -> tuple[bool, str]:
    f = free_mib(0)
    if f < need_mib:
        return False, f"card 0 has {f} MiB free, needs {need_mib}"
    w = waiting()
    if w:
        return False, f"trainer job waiting for a card: {w}"
    return True, f"card 0: {f} MiB free, nothing waiting"


if __name__ == "__main__":
    ok, why = ready()
    print(why)
    raise SystemExit(0 if ok else 1)
