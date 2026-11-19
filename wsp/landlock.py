"""A file-system allow-list for a programmer's sandbox (Landlock, Linux >= 5.13; docs/TROCR_STAGE4_PLAN.md section 7, item 2).

Until stage 3 the donor-free arm was enforced by the brief only: the sandbox (``unshare -Un``, ``env -i``) cuts the network but every file of the
user stays readable, donor weights, other sessions' folders and the held-out lines included. ``restrict`` limits the calling process (and everything
it starts) to the paths it is given; the restriction cannot be lifted from inside. No privilege and no mount namespace is needed (``unshare -Um`` is
refused on this host). ``session_paths`` computes the allow-list of one session; ``visible_weights`` lists the weight files a process can open,
which is what ``tests/test_landlock.py`` and the stage-4 harness check print.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

_SYS_CREATE, _SYS_ADD, _SYS_RESTRICT = 444, 445, 446
(EXECUTE, WRITE_FILE, READ_FILE, READ_DIR, REMOVE_DIR, REMOVE_FILE, MAKE_CHAR, MAKE_DIR, MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM,
 REFER, TRUNCATE, IOCTL_DEV) = (1 << i for i in range(16))
_FILE_BITS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV
_READ = EXECUTE | READ_FILE | READ_DIR


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def abi_version() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    v = libc.syscall(_SYS_CREATE, None, 0, 1)             # LANDLOCK_CREATE_RULESET_VERSION
    return int(v) if v > 0 else 0


def _handled(abi: int) -> int:
    m = (1 << 13) - 1                                       # bits 0..12 (ABI 1)
    if abi >= 2:
        m |= REFER
    if abi >= 3:
        m |= TRUNCATE
    if abi >= 5:
        m |= IOCTL_DEV
    return m


def _mounts_beneath(paths: list[str | Path]) -> list[str]:
    """Mount points inside the given paths: a Landlock rule does not cross a mount (a ``/dev/nvidia-caps`` of its own, ``/sys/fs/cgroup``), so each gets the
    rule of the path that contains it."""
    mps = []
    try:
        for line in open("/proc/self/mountinfo"):
            mps.append(line.split()[4].encode().decode("unicode_escape"))
    except OSError:
        return []
    out = []
    for p in paths:
        p = str(p).rstrip("/") + "/"
        out += [m for m in mps if m.rstrip("/") + "/" != p and (m.rstrip("/") + "/").startswith(p)]
    return out


def restrict(read: list[str | Path], write: list[str | Path] = ()) -> dict:
    """Limit this process to read access under ``read`` and read/write access under ``write``. Irreversible. Missing paths are skipped (reported)."""
    libc = ctypes.CDLL(None, use_errno=True)
    abi = abi_version()
    if abi < 1:
        raise OSError("Landlock is not available on this kernel")
    handled = _handled(abi)
    attr = ctypes.c_uint64(handled)
    fd = libc.syscall(_SYS_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    skipped = []
    read, write = list(read), list(write)
    read = read + _mounts_beneath(read)
    write = write + _mounts_beneath(write)
    for paths, access in ((read, _READ), (write, handled)):
        for p in paths:
            p = str(p)
            try:
                pfd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                skipped.append(p)
                continue
            try:
                acc = access & handled
                if not os.path.isdir(p):
                    acc &= _FILE_BITS
                rule = _PathBeneath(acc, pfd)
                if libc.syscall(_SYS_ADD, fd, 1, ctypes.byref(rule), 0) < 0:   # LANDLOCK_RULE_PATH_BENEATH
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {p}")
            finally:
                os.close(pfd)
    if libc.prctl(38, 1, 0, 0, 0) < 0:                      # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS) failed")
    if libc.syscall(_SYS_RESTRICT, fd, 0) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(fd)
    return {"abi": abi, "read": [str(p) for p in read], "write": [str(p) for p in write], "skipped": skipped}


SYSTEM_READ = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys"]
SYSTEM_WRITE = ["/dev", "/proc"]                       # /proc: CUDA names its threads (/proc/self/task/*/comm); without it cudaGetDeviceCount fails (error 304)


def session_paths(run_dir: Path, subject: Path, practice_list: Path, python: Path, repo: Path, donors: list[Path] = (), tmp: Path | None = None) -> dict:
    """The allow-list of a TrOCR session: its interpreter and venv, the harness package, the subject, the practice list and its images, its own run folder
    (read and write) and the temporary directory; the donors only when the arm has them. Nothing else: no other model, no other run, no held-out line."""
    py = Path(python)
    venv = py.parent.parent
    read = list(SYSTEM_READ) + [str(venv), str(Path(repo) / "wsp"), str(subject), str(practice_list)]
    try:                                                     # the interpreter the venv points to (uv-managed or system)
        read.append(str(Path(os.path.realpath(py)).parent.parent))
    except OSError:
        pass
    imgs = {str(Path(json.loads(l)["image"]).parent) for l in Path(practice_list).read_text().splitlines()[:5000] if l.strip()}
    read += sorted(imgs)
    read += [str(Path.home() / "atr-cache" / "wsp" / "card-state.json")]   # the card gate (wsp.card.ready) reads the watcher's state file
    read += [str(d) for d in donors]
    write = list(SYSTEM_WRITE) + [str(run_dir)] + ([str(tmp)] if tmp else [])
    return {"read": read, "write": write}


def visible_weights(roots: list[str | Path], names: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors"), max_depth: int = 6) -> list[str]:
    """The weight files under ``roots`` that this process can open for reading."""
    out = []
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        base = len(r.parts)
        for dp, dn, fn in os.walk(r, onerror=lambda e: None):
            if len(Path(dp).parts) - base >= max_depth:
                dn[:] = []
            for f in fn:
                if f in names or f.endswith(".safetensors"):
                    p = Path(dp) / f
                    try:
                        with open(p, "rb") as fh:
                            fh.read(1)
                        out.append(str(p))
                    except OSError:
                        pass
    return sorted(out)


def openable(paths: list[str | Path]) -> list[str]:
    """Of the given files, those this process can open for reading. A restricted process cannot list the directories above its allowed paths, so the check
    is made on candidate paths found beforehand (``visible_weights`` outside the sandbox) rather than by walking."""
    out = []
    for p in paths:
        try:
            with open(p, "rb") as fh:
                fh.read(1)
            out.append(str(p))
        except OSError:
            pass
    return sorted(out)


if __name__ == "__main__":                                 # python -m wsp.landlock READ_PATH... : self-test, prints what stays visible
    restrict(sys.argv[1:])
    print(json.dumps(visible_weights(["/tmp"]), indent=1))
