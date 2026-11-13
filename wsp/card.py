"""Card 0 before a pass: enough memory, and no trainer job waiting for it (#396).

The trainer's ``GET /health`` carries ``waiting`` (``null`` or
``{"job_id", "engine", "needs_mib"}``). A job waiting while card 0 is ours is
a job we would hold back, so the pass does not start. Card 1 is never used.

Inside the sandbox (``unshare -Un``) loopback is cut and the trainer cannot be
asked. ``python -m wsp.card --watch`` (or ``watch``), run outside it, writes the answer to
``STATE`` once a minute; ``waiting`` reads that file when the trainer is out of
reach, and refuses (no pass) when the file is older than three minutes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HEALTH = "http://127.0.0.1:8204/health"
from wsp.vlm_reference import HOME as _HOME
STATE = _HOME / "atr-cache/wsp/card-state.json"   # ~ from the repo's place, as wsp.vlm_reference
FRESH_S = 180


def free_mib(card: int = 0) -> int:
    out = subprocess.run(["nvidia-smi", f"--id={card}", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def _ask(timeout: float = 90.0):
    with urllib.request.urlopen(HEALTH, timeout=timeout) as r:   # /health takes ~40 s on asteraix
        return json.load(r).get("waiting")


def waiting(timeout: float = 90.0):
    try:
        return _ask(timeout)
    except (urllib.error.URLError, OSError):
        if not STATE.exists():
            raise RuntimeError("trainer out of reach and no watcher state (run `python -m wsp.card --watch`)")
        st = json.loads(STATE.read_text())
        age = time.time() - st["time"]
        if age > FRESH_S:
            raise RuntimeError(f"trainer out of reach and the watcher state is {age:.0f} s old")
        return st["waiting"]


def watch(interval: float = 60.0) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            w = _ask()
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"time": time.time(), "waiting": w}))
            tmp.replace(STATE)
        except Exception as e:                   # a failed ask leaves the file to go stale: fail closed
            print(time.strftime("%H:%M"), "ask failed:", e, file=sys.stderr, flush=True)
        time.sleep(interval)


def ready(need_mib: int = 12000) -> tuple[bool, str]:
    f = free_mib(0)
    if f < need_mib:
        return False, f"card 0 has {f} MiB free, needs {need_mib}"
    try:
        w = waiting()
    except RuntimeError as e:
        return False, str(e)
    if w:
        return False, f"trainer job waiting for a card: {w}"
    return True, f"card 0: {f} MiB free, nothing waiting"


if __name__ == "__main__":
    if "--watch" in sys.argv or sys.argv[1:] == ["watch"]:   # `watch` is the spelling of docs/ADAPTER_REPORT.md
        watch()
    ok, why = ready()
    print(why)
    raise SystemExit(0 if ok else 1)
