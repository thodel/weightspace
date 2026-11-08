"""Card 0 before a pass: enough memory, and no trainer job waiting for it (#396).

The trainer's ``GET /health`` carries ``waiting`` (``null`` or
``{"job_id", "engine", "needs_mib"}``). A job waiting while card 0 is ours is
a job we would hold back, so the pass does not start. Card 1 is never used.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

HEALTH = "http://127.0.0.1:8204/health"


def free_mib(card: int = 0) -> int:
    out = subprocess.run(["nvidia-smi", f"--id={card}", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def _home() -> Path:
    """The real home: inside ``unshare -Un`` the uid maps to nobody and HOME is the run folder, so take it from this file's path."""
    parts = Path(__file__).resolve().parts
    return Path(*parts[:3]) if parts[:2] == ("/", "home") else Path.home()


FLAG = _home() / "atr-cache" / "wsp" / "spool" / "TRAINER_WAITING"


def waiting(timeout: float = 90.0):
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as r:   # /health takes ~40 s on asteraix
            return json.load(r).get("waiting")
    except OSError:
        # inside the session sandbox (unshare -Un) loopback is down: read the flag a watcher outside keeps
        # (``python -m wsp.card watch``); no flag file means the watcher says nothing is waiting
        if not (FLAG.parent / "WATCHER_ALIVE").exists():
            raise
        return FLAG.read_text() if FLAG.exists() else None


def ready(need_mib: int = 12000) -> tuple[bool, str]:
    f = free_mib(0)
    if f < need_mib:
        return False, f"card 0 has {f} MiB free, needs {need_mib}"
    w = waiting()
    if w:
        return False, f"trainer job waiting for a card: {w}"
    return True, f"card 0: {f} MiB free, nothing waiting"


def watch(every: float = 240.0):
    """Outside the sandbox: keep FLAG in step with the trainer's ``waiting`` (and WATCHER_ALIVE fresh)."""
    import time
    while True:
        try:
            w = waiting()
            (FLAG.write_text(json.dumps(w)) if w else FLAG.unlink(missing_ok=True))
            (FLAG.parent / "WATCHER_ALIVE").write_text(str(time.time()))
        except Exception as e:  # health down: leave the flag as it was, drop liveness so passes refuse
            (FLAG.parent / "WATCHER_ALIVE").unlink(missing_ok=True)
            print("watch:", e, flush=True)
        time.sleep(every)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["watch"]:
        watch()
    ok, why = ready()
    print(why)
    raise SystemExit(0 if ok else 1)
