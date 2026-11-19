"""An external language model (Gemini, GPT) as the programmer of a weight-space session.

The Fable replicates gave the programmer role to a Claude Code subagent with the tools Read,
Write, Edit and Bash. Models from other vendors have no such harness, so this module gives them
the same reach through function calls:

- ``read_file`` and ``list_dir``: only the paths that the run's ``programmer.json`` allows,
  which are the paths the brief lists. With Fable this rule held only because the brief said
  so; here it is enforced as well.
- ``write_file``: only ``preregistration.md`` (§5 alone may change), ``program_NN.py`` and
  ``notes.md`` in the run folder.
- ``run_python``: CPU-only reading code. It runs in the same sandbox as the programs
  (``env -i``, ``unshare -Un``, ``CUDA_VISIBLE_DEVICES=``), but on a throw-away copy of the run
  folder, so a stray ``measure`` or ``keep`` cannot touch the session.
- ``finish_round``: the model says it has finished the round.

The operator drives each round, in the same order as with Fable:

    python -m wsp.programmer check  --provider gemini
    python -m wsp.programmer round  --run runs/<id> --provider gemini --round 0
    (operator runs program_01.py in the sandbox, stdout to stdout_program_01.log)
    python -m wsp.programmer round  --run runs/<id> --provider gemini --round 1 --stdout runs/<id>/stdout_program_01.log

The conversation carries over from round to round, as the Fable subagent's did. It is kept in
``<run>/programmer/``, which holds ``conversation.json`` (the provider's own format, re-sent as
is, Gemini's thought signatures included), ``transcript.jsonl`` (every call and tool result)
and ``usage.json``. API keys come from the environment or from ``~/atr-cache/wsp/keys/`` and
are never written anywhere.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEY_DIR = Path(os.environ.get("WSP_KEY_DIR", Path.home() / "atr-cache" / "wsp" / "keys"))
TMP = Path(os.environ.get("TMPDIR", Path.home() / "atr-cache" / "tmp"))

PROVIDERS = {
    # The model id is pinned in programmer.json for each run. These are only defaults for `check`.
    "gemini": {"env": "GEMINI_API_KEY", "key_file": "gemini.key", "model_env": "WSP_GEMINI_MODEL",
               "default_model": "gemini-4-argon",
               "base": "https://generativelanguage.googleapis.com/v1beta"},
    "openai": {"env": "OPENAI_API_KEY", "key_file": "openai.key", "model_env": "WSP_OPENAI_MODEL",
               "default_model": None,  # set WSP_OPENAI_MODEL or --model; `check` lists the choices
               "base": "https://api.openai.com/v1"},
}

WRITABLE = [r"preregistration\.md", r"program_\d\d\.py", r"notes\.md"]
SCRATCH_IGNORE = shutil.ignore_patterns("passes", "programmer", "*.adapter", "work.adapter",
                                        "*.wsp-work", "*.mlmodel", "export", "attempt*")
MAX_READ_CHARS = 200_000
MAX_OUT_CHARS = 30_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- keys

def api_key(provider: str) -> str:
    p = PROVIDERS[provider]
    key = os.environ.get(p["env"], "").strip()
    if key:
        return key
    f = KEY_DIR / p["key_file"]
    if f.exists():
        if f.stat().st_mode & 0o077:
            raise PermissionError(f"{f} is readable by others; run: chmod 600 {f}")
        return f.read_text().strip()
    raise KeyError(f"no API key for {provider}: set {p['env']} or write it to {f} (chmod 600)")


def redact(text: str, key: str | None) -> str:
    return text.replace(key, "***") if key else text


# ---------------------------------------------------------------- HTTP

def post_json(url: str, body: dict, headers: dict, key: str, tries: int = 6, timeout: int = 900) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = redact(e.read().decode(errors="replace")[:2000], key)
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(min(300, 10 * 2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {e.code} from {redact(url, key)}: {msg}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < tries - 1:
                time.sleep(min(300, 10 * 2 ** attempt))
                continue
            raise RuntimeError(f"no answer from {redact(url, key)}: {e}") from None
    raise AssertionError("unreachable")


def get_json(url: str, headers: dict, key: str) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {redact(e.read().decode(errors='replace')[:2000], key)}") from None


# ---------------------------------------------------------------- tools (provider-neutral)

TOOLS = [
    {"name": "read_file",
     "description": "Read a text file. Path relative to the repository root (e.g. wsp/ops.py or "
                    "runs/<id>/log.jsonl). Only the files your brief allows are readable. Lines are numbered.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"},
         "offset": {"type": "integer", "description": "first line, 1-based (default 1)"},
         "limit": {"type": "integer", "description": "number of lines (default 2000)"}},
         "required": ["path"]}},
    {"name": "list_dir",
     "description": "List a directory that your brief allows you to read, relative to the repository root.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file",
     "description": "Write a whole file in your run folder. Allowed: preregistration.md (only section 5 "
                    "may change), program_NN.py, notes.md. Path relative to the repository root.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "run_python",
     "description": "Run a CPU-only Python reading script with the harness's interpreter (wsp and numpy), "
                    "in the sandbox, with the working directory set to a throw-away copy of your run folder. "
                    "No GPU, no network; nothing it writes is kept. Use it to read the subject through the API. "
                    "Never call measure, keep or freeze here and never run program_*.py. Returns stdout and stderr.",
     "parameters": {"type": "object", "properties": {
         "code": {"type": "string"},
         "timeout_s": {"type": "integer", "description": "default 600, at most 1800"}}, "required": ["code"]}},
    {"name": "finish_round",
     "description": "Call this once you have finished this round: your files are written and you wait for "
                    "the operator. Give a short summary of what you wrote.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

FORBIDDEN_IN_SCRATCH = re.compile(r"\.(measure|keep|freeze)\s*\(|program_\d\d|heldout|subprocess|os\.system")


@dataclass
class Workspace:
    """What the programmer may touch, from <run>/programmer.json."""
    root: Path                 # the repository checkout the run lives in
    run_dir: Path              # absolute
    readable: list[str]        # glob patterns, relative to root
    python: Path

    @classmethod
    def load(cls, run_dir: Path) -> "Workspace":
        run_dir = run_dir.resolve()
        cfg = json.loads((run_dir / "programmer.json").read_text())
        root = Path(cfg.get("root", REPO)).resolve()
        rel = run_dir.relative_to(root).as_posix()
        readable = [p.replace("{run}", rel) for p in cfg["readable"]] + [f"{rel}/*"]
        return cls(root, run_dir, readable, Path(cfg.get("python", root / ".venv/bin/python")))

    def _resolve(self, path: str) -> tuple[Path, str]:
        p = (self.root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        try:
            rel = p.relative_to(self.root).as_posix()
        except ValueError:
            raise PermissionError(f"{path} is outside the repository")
        return p, rel

    def may_read(self, rel: str) -> bool:
        own = self.run_dir.relative_to(self.root).as_posix() + "/programmer"
        if rel == own or rel.startswith(own + "/"):
            return False  # its own transcript is the operator's record, not reading matter
        return any(fnmatch.fnmatch(rel, g) for g in self.readable)  # fnmatch's * also matches "/"

    def may_list(self, rel: str) -> bool:
        rel = rel.strip("/")
        if rel in ("", "."):
            return True
        own = self.run_dir.relative_to(self.root).as_posix() + "/programmer"
        if rel == own or rel.startswith(own + "/"):
            return False
        return any(g.startswith(rel + "/") or fnmatch.fnmatch(rel + "/x", g) for g in self.readable)

    # -- the tools

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p, rel = self._resolve(path)
        except PermissionError as e:
            return f"refused: {e}"
        if not self.may_read(rel):
            return f"refused: {rel} is not on the list of files your brief allows"
        if not p.is_file():
            return f"no such file: {rel}"
        lines = p.read_text(errors="replace").splitlines()
        start = max(1, int(offset or 1))
        chunk = lines[start - 1:start - 1 + max(1, int(limit or 2000))]
        out = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(chunk, start))
        more = len(lines) - (start - 1 + len(chunk))
        if len(out) > MAX_READ_CHARS:
            out = out[:MAX_READ_CHARS] + "\n[cut: read a smaller range with offset/limit]"
        return out + (f"\n[{more} more lines; use offset]" if more > 0 else "")

    def list_dir(self, path: str) -> str:
        try:
            p, rel = self._resolve(path)
        except PermissionError as e:
            return f"refused: {e}"
        if not (p.is_dir() and self.may_list(rel)):
            return f"refused or not a directory: {rel}"
        names = []
        for c in sorted(p.iterdir()):
            crel = c.relative_to(self.root).as_posix()
            if c.is_dir() and self.may_list(crel) or c.is_file() and self.may_read(crel):
                names.append(c.name + ("/" if c.is_dir() else ""))
        return "\n".join(names) or "(nothing readable here)"

    def write_file(self, path: str, content: str) -> str:
        try:
            p, rel = self._resolve(path)
        except PermissionError as e:
            return f"refused: {e}"
        if p.parent != self.run_dir or not any(re.fullmatch(w, p.name) for w in WRITABLE):
            return f"refused: you may write only preregistration.md, program_NN.py and notes.md in {self.run_dir.name}"
        if p.name == "preregistration.md" and p.exists():
            old, new = outside_section5(p.read_text()), outside_section5(content)
            if old != new:
                return "refused: only section 5 of preregistration.md may change; the rest must stay as it is"
        p.write_text(content)
        if p.suffix == ".py":
            r = subprocess.run([str(self.python), "-m", "py_compile", str(p)], capture_output=True, text=True)
            if r.returncode:
                return f"written ({len(content)} chars), but py_compile fails:\n{r.stderr[-3000:]}"
        return f"written: {rel} ({len(content)} chars)"

    def run_python(self, code: str, timeout_s: int = 600) -> str:
        if FORBIDDEN_IN_SCRATCH.search(code):
            return ("refused: reading scripts must not call measure/keep/freeze, run programs, score held-out "
                    "or start processes")
        timeout_s = max(10, min(int(timeout_s or 600), 1800))
        TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="wsp-scratch-", dir=TMP) as td:
            work = Path(td) / self.run_dir.name
            shutil.copytree(self.run_dir, work, ignore=SCRATCH_IGNORE)
            (work / "_scratch.py").write_text(code)
            env = ["env", "-i", "PATH=/usr/bin:/bin", f"HOME={work}", f"TMPDIR={TMP}", "CUDA_VISIBLE_DEVICES=",
                   "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                   "no_proxy=*"]
            boot = (f"import sys,runpy; sys.path.insert(0, {str(self.root)!r}); "
                    "runpy.run_path('_scratch.py', run_name='__main__')")
            cmd = env + (["unshare", "-Un"] if shutil.which("unshare") else []) + [str(self.python), "-I", "-c", boot]
            try:
                r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                return f"timeout after {timeout_s} s"
        out = f"exit {r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}"
        return out if len(out) <= MAX_OUT_CHARS else out[:MAX_OUT_CHARS // 2] + "\n[...cut...]\n" + out[-MAX_OUT_CHARS // 2:]

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "read_file":
                return self.read_file(args["path"], args.get("offset", 1), args.get("limit", 2000))
            if name == "list_dir":
                return self.list_dir(args["path"])
            if name == "write_file":
                return self.write_file(args["path"], args["content"])
            if name == "run_python":
                return self.run_python(args["code"], args.get("timeout_s", 600))
            return f"unknown tool {name}"
        except (KeyError, TypeError) as e:
            return f"bad arguments for {name}: {e}"
        except PermissionError as e:
            return f"refused: {e}"


def outside_section5(text: str) -> str:
    """The pre-registration with section 5 cut out: from the line that starts "5. " (or "## 5")
    to the next numbered section."""
    return re.sub(r"(?ms)^(?:## )?5\. .*?(?=^\d\. \*\*|^## |\Z)", "5.\n", text)


# ---------------------------------------------------------------- providers

@dataclass
class Turn:
    text: str
    calls: list[tuple[str, str, dict]]   # (call id, name, args)
    usage: dict
    raw: dict


class OpenAIResponses:
    """The Responses API with function tools: the default for OpenAI, because current GPT models
    refuse function tools on Chat Completions unless reasoning is switched off. Output items
    (reasoning, messages, function calls) go back unchanged in the next request."""

    def __init__(self, model: str, key: str, base: str, options: dict | None = None):
        self.model, self.key, self.base, self.options = model, key, base.rstrip("/"), options or {}

    def start(self, system: str) -> list:
        return [{"_system": system}]

    def add_user(self, conv: list, text: str) -> None:
        conv.append({"role": "user", "content": text})

    def send(self, conv: list) -> Turn:
        body = {"model": self.model, "instructions": conv[0]["_system"], "input": conv[1:],
                "tools": [{"type": "function", **t} for t in TOOLS], **self.options}
        r = post_json(f"{self.base}/responses", body, {"Authorization": f"Bearer {self.key}"}, self.key)
        text, calls = [], []
        for item in r.get("output", []):
            conv.append(item)
            if item.get("type") == "function_call":
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_unparsed": item.get("arguments")}
                calls.append((item["call_id"], item["name"], args))
            elif item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
        u = r.get("usage", {})
        return Turn("\n".join(text), calls,
                    {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                     "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)},
                    {"id": r.get("id"), "model": r.get("model"), "finish": r.get("status")})

    def add_results(self, conv: list, results: list[tuple[str, str, str]]) -> None:
        for cid, _name, out in results:
            conv.append({"type": "function_call_output", "call_id": cid, "output": out})

    def models(self) -> list[str]:
        r = get_json(f"{self.base}/models", {"Authorization": f"Bearer {self.key}"}, self.key)
        return sorted(m["id"] for m in r.get("data", []))


class OpenAIChat:
    """Chat Completions with function tools, for compatible servers or older models (options {"api": "chat"})."""

    def __init__(self, model: str, key: str, base: str, options: dict | None = None):
        self.model, self.key, self.base, self.options = model, key, base.rstrip("/"), options or {}

    def start(self, system: str) -> list:
        return [{"role": "system", "content": system}]

    def add_user(self, conv: list, text: str) -> None:
        conv.append({"role": "user", "content": text})

    def send(self, conv: list) -> Turn:
        body = {"model": self.model, "messages": conv,
                "tools": [{"type": "function", "function": t} for t in TOOLS], **self.options}
        r = post_json(f"{self.base}/chat/completions", body, {"Authorization": f"Bearer {self.key}"}, self.key)
        msg = r["choices"][0]["message"]
        conv.append({k: v for k, v in msg.items() if v is not None and k not in ("refusal", "annotations")})
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": c["function"]["arguments"]}
            calls.append((c["id"], c["function"]["name"], args))
        u = r.get("usage", {})
        return Turn(msg.get("content") or "", calls,
                    {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0),
                     "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)},
                    {"id": r.get("id"), "model": r.get("model"), "finish": r["choices"][0].get("finish_reason")})

    def add_results(self, conv: list, results: list[tuple[str, str, str]]) -> None:
        for cid, _name, out in results:
            conv.append({"role": "tool", "tool_call_id": cid, "content": out})

    def models(self) -> list[str]:
        r = get_json(f"{self.base}/models", {"Authorization": f"Bearer {self.key}"}, self.key)
        return sorted(m["id"] for m in r.get("data", []))


class Gemini:
    """generateContent with function declarations. The model's content goes back unchanged, thought
    signatures included, as the API requires for multi-turn function calling."""

    def __init__(self, model: str, key: str, base: str, options: dict | None = None):
        self.model, self.key, self.base, self.options = model, key, base.rstrip("/"), options or {}
        self.system = ""

    def start(self, system: str) -> list:
        self.system = system
        return [{"_system": system}]

    def add_user(self, conv: list, text: str) -> None:
        conv.append({"role": "user", "parts": [{"text": text}]})

    def send(self, conv: list) -> Turn:
        system = conv[0].get("_system", self.system)
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": conv[1:],
                "tools": [{"functionDeclarations": TOOLS}], **self.options}
        url = f"{self.base}/models/{self.model}:generateContent"
        r = post_json(url, body, {"x-goog-api-key": self.key}, self.key)
        cand = (r.get("candidates") or [{}])[0]
        content = cand.get("content") or {"role": "model", "parts": []}
        content.setdefault("role", "model")
        conv.append(content)
        text, calls = [], []
        for i, part in enumerate(content.get("parts", [])):
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append((fc.get("id") or f"call_{i}", fc["name"], fc.get("args") or {}))
            elif "text" in part and not part.get("thought"):
                text.append(part["text"])
        u = r.get("usageMetadata", {})
        return Turn("\n".join(text), calls,
                    {"input": u.get("promptTokenCount", 0), "output": u.get("candidatesTokenCount", 0),
                     "reasoning": u.get("thoughtsTokenCount", 0)},
                    {"model": r.get("modelVersion"), "finish": cand.get("finishReason"),
                     "id": r.get("responseId")})

    def add_results(self, conv: list, results: list[tuple[str, str, str]]) -> None:
        parts = []
        for cid, name, out in results:
            fr = {"name": name, "response": {"result": out}}
            if not cid.startswith("call_"):
                fr["id"] = cid
            parts.append({"functionResponse": fr})
        conv.append({"role": "user", "parts": parts})

    def models(self) -> list[str]:
        r = get_json(f"{self.base}/models?pageSize=1000", {"x-goog-api-key": self.key}, self.key)
        return sorted(m["name"].removeprefix("models/") for m in r.get("models", []))


def client(provider: str, model: str | None = None, options: dict | None = None):
    p = PROVIDERS[provider]
    model = model or os.environ.get(p["model_env"]) or p["default_model"]
    if not model:
        raise KeyError(f"no model for {provider}: pass --model or set {p['model_env']}")
    base = os.environ.get(f"WSP_{provider.upper()}_BASE", p["base"])
    options = dict(options or {})
    api = options.pop("api", "responses")  # openai only: "responses" (default) or "chat"
    cls = Gemini if provider == "gemini" else (OpenAIChat if api == "chat" else OpenAIResponses)
    return cls(model, api_key(provider), base, options)


# ---------------------------------------------------------------- the round loop

SYSTEM = """You are the programmer of a weight-space session. You work only through the tools you are given.
Your instructions are in the file PROGRAMMER_BRIEF.md in your run folder; read it first and follow it exactly.
Tool equivalents of what the brief says:
- "read" a file: read_file / list_dir (paths relative to the repository root; only the files the brief lists are open to you);
- "write" a file: write_file (whole file; only preregistration.md section 5, program_NN.py and notes.md in your run folder);
- run a CPU reading script or py_compile: run_python (wsp and numpy, CPU only, on a throw-away copy of the run folder;
  write_file already checks program files with py_compile).
You never run program_*.py; the operator does, and gives you its stdout at the start of the next round.
When a round's files are written, call finish_round. Do not stop without calling it."""


def round_prompt(run_rel: str, n: int, stdout_text: str | None, model: str) -> str:
    if n == 0:
        return (f"Your run folder is {run_rel}/. Read {run_rel}/PROGRAMMER_BRIEF.md and do round 0 as it describes. "
                f"In notes.md give your model id as `{model}` and your settings as given by the operator.")
    return (f"The operator ran {run_rel}/program_{n:02d}.py in the sandbox. Its stdout follows. "
            f"Read log.jsonl and passes/ as the brief allows, append to notes.md, and write "
            f"program_{n + 1:02d}.py (end with s.freeze() if you are done or the budget is used up). "
            f"Then call finish_round.\n\n--- stdout of program_{n:02d}.py ---\n{stdout_text or '(empty)'}")


@dataclass
class Recorder:
    folder: Path
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "reasoning": 0, "calls": 0})

    def __post_init__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        f = self.folder / "usage.json"
        if f.exists():
            self.usage = json.loads(f.read_text())

    def log(self, **event) -> None:
        with open(self.folder / "transcript.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": now(), **event}, ensure_ascii=False) + "\n")

    def add_usage(self, u: dict) -> None:
        for k in ("input", "output", "reasoning"):
            self.usage[k] = self.usage.get(k, 0) + int(u.get(k) or 0)
        self.usage["calls"] = self.usage.get("calls", 0) + 1
        (self.folder / "usage.json").write_text(json.dumps(self.usage, indent=1))


def run_round(run_dir: Path, provider: str, n: int, stdout_file: Path | None = None,
              max_steps: int = 200, llm=None) -> dict:
    ws = Workspace.load(run_dir)
    cfg = json.loads((ws.run_dir / "programmer.json").read_text())
    if cfg.get("provider") != provider:
        raise ValueError(f"programmer.json pins provider {cfg.get('provider')!r}, not {provider!r}")
    llm = llm or client(provider, cfg["model"], cfg.get("options"))
    rec = Recorder(ws.run_dir / "programmer")
    conv_file = rec.folder / "conversation.json"
    done_rounds = json.loads((rec.folder / "rounds.json").read_text()) if (rec.folder / "rounds.json").exists() else []
    if n != len(done_rounds):
        raise ValueError(f"round {n} requested, but {len(done_rounds)} rounds are recorded; next is {len(done_rounds)}")
    conv = json.loads(conv_file.read_text()) if conv_file.exists() else llm.start(SYSTEM)
    run_rel = ws.run_dir.relative_to(ws.root).as_posix()
    prompt = round_prompt(run_rel, n, stdout_file.read_text() if stdout_file else None, cfg["model"])
    llm.add_user(conv, prompt)
    rec.log(round=n, role="operator", text=prompt)
    finished, summary, steps, nudged = False, "", 0, False
    while steps < max_steps:
        steps += 1
        turn = llm.send(conv)
        rec.add_usage(turn.usage)
        rec.log(round=n, role="model", text=turn.text, calls=[[c[1], c[2]] for c in turn.calls],
                usage=turn.usage, meta=turn.raw)
        conv_file.write_text(json.dumps(conv, ensure_ascii=False))
        if not turn.calls:
            if nudged:
                break
            llm.add_user(conv, "Continue the round with your tools, and call finish_round when its files are written.")
            nudged = True
            continue
        results = []
        for cid, name, args in turn.calls:
            if name == "finish_round":
                finished, summary = True, str(args.get("summary", ""))
                out = "round closed; wait for the operator"
            else:
                out = ws.call(name, args)
            rec.log(round=n, role="tool", name=name, args=_short_args(args), result=out[:4000])
            results.append((cid, name, out))
        llm.add_results(conv, results)
        conv_file.write_text(json.dumps(conv, ensure_ascii=False))
        if finished:
            break
    expected = ws.run_dir / f"program_{n + 1:02d}.py"
    record = {"round": n, "finished": finished, "steps": steps, "summary": summary, "ended": now(),
              "program": expected.name if expected.exists() else None, "usage_total": rec.usage}
    done_rounds.append(record)
    (rec.folder / "rounds.json").write_text(json.dumps(done_rounds, indent=1))
    return record


def _short_args(args: dict) -> dict:
    return {k: (v if not isinstance(v, str) or len(v) <= 20000 else v[:20000] + "…[cut]") for k, v in args.items()}


# ---------------------------------------------------------------- CLI

def cmd_check(provider: str, model: str | None) -> int:
    """Key present, model listed, one tool call answered. Prints no key."""
    llm = client(provider, model)
    print(f"{provider}: key found; model {llm.model}; base {llm.base}")
    try:
        ids = llm.models()
        print(f"  {len(ids)} models listed; pinned model {'listed' if llm.model in ids else 'NOT listed'}")
        if llm.model not in ids:
            print("  candidates:", ", ".join(i for i in ids if re.search(r"gpt|gemini|o\d", i))[:2000])
    except RuntimeError as e:
        print(f"  model list failed: {e}")
    conv = llm.start("Answer only by calling the tool finish_round.")
    llm.add_user(conv, "Call finish_round with the summary 'ok'.")
    t = llm.send(conv)
    ok = any(c[1] == "finish_round" for c in t.calls)
    print(f"  tool call: {'ok' if ok else 'NO tool call: ' + t.text[:200]}; served model {t.raw.get('model')}; usage {t.usage}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wsp.programmer", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--provider", choices=PROVIDERS, required=True); c.add_argument("--model")
    r = sub.add_parser("round"); r.add_argument("--run", required=True, type=Path)
    r.add_argument("--provider", choices=PROVIDERS, required=True); r.add_argument("--round", type=int, required=True)
    r.add_argument("--stdout", type=Path); r.add_argument("--max-steps", type=int, default=200)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            return cmd_check(a.provider, a.model)
    except (KeyError, PermissionError) as e:
        print(f"not ready: {e}", file=sys.stderr)
        return 3
    if a.round > 0 and not a.stdout:
        ap.error("--stdout is required from round 1 on")
    rec = run_round(a.run, a.provider, a.round, a.stdout, a.max_steps)
    print(json.dumps(rec, indent=1))
    return 0 if rec["finished"] else 2


if __name__ == "__main__":
    sys.exit(main())
