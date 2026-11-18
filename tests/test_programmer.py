"""wsp.programmer without network: a scripted model drives the tool loop over a fake checkout."""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from wsp import programmer as P


def make_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    run = root / "runs" / "20261115-x-g1-b10-1"
    (run / "passes").mkdir(parents=True)
    (root / "wsp").mkdir()
    (root / "docs" / "secret").mkdir(parents=True)
    (root / "wsp" / "ops.py").write_text("def scale(): pass\n")
    (root / "wsp" / "zoo.py").write_text("# not on the list\n")
    (root / "docs" / "secret" / "REPORT.md").write_text("the earlier session kept H1\n")
    (root / "docs" / "reading.md").write_text("line 1\nline 2\nline 3\n")
    (run / "PROGRAMMER_BRIEF.md").write_text("brief\n")
    (run / "passes" / "pass-001.json").write_text("{}\n")
    (run / "preregistration.md").write_text(
        "# Pre-registration\n\n4. **Budget:** 10.\n\n5. **Hypotheses**\n\n   _To be written._\n\n6. **Controls.** C0.\n")
    (run / "programmer.json").write_text(json.dumps({
        "provider": "gemini", "model": "fake-model", "root": str(root), "python": sys.executable,
        "readable": ["docs/reading.md", "wsp/ops.py"]}))
    return root, run


def test_read_rules(tmp_path):
    root, run = make_repo(tmp_path)
    ws = P.Workspace.load(run)
    assert "line 2" in ws.read_file("docs/reading.md")
    assert ws.read_file("docs/reading.md", offset=3, limit=1).strip().endswith("line 3")
    assert ws.read_file("docs/secret/REPORT.md").startswith("refused")
    assert ws.read_file("wsp/zoo.py").startswith("refused")
    assert ws.read_file("../outside.txt").startswith("refused")
    assert ws.read_file(f"{run.relative_to(root)}/passes/pass-001.json").strip().endswith("{}")
    (run / "programmer").mkdir()
    (run / "programmer" / "transcript.jsonl").write_text("x\n")
    assert ws.read_file(f"{run.relative_to(root)}/programmer/transcript.jsonl").startswith("refused")
    top = ws.list_dir(".")
    assert "wsp/" in top and "docs/" in top and "runs/" in top
    assert "secret" not in ws.list_dir("docs") and "reading.md" in ws.list_dir("docs")
    assert "zoo.py" not in ws.list_dir("wsp")
    assert "programmer/" not in ws.list_dir(str(run.relative_to(root)))


def test_write_rules(tmp_path):
    root, run = make_repo(tmp_path)
    ws = P.Workspace.load(run)
    rel = run.relative_to(root)
    assert ws.write_file(f"{rel}/program_01.py", "print('hi')\n").startswith("written")
    assert "py_compile fails" in ws.write_file(f"{rel}/program_02.py", "def (:\n")
    assert ws.write_file(f"{rel}/other.py", "x").startswith("refused")
    assert ws.write_file("wsp/ops.py", "x").startswith("refused")
    pre = (run / "preregistration.md").read_text()
    ok = pre.replace("_To be written._", "- **H1:** project out dir 0; CER -3 points.")
    assert ws.write_file(f"{rel}/preregistration.md", ok).startswith("written")
    bad = ok.replace("10.", "20.")
    assert "only section 5" in ws.write_file(f"{rel}/preregistration.md", bad)
    assert (run / "preregistration.md").read_text() == ok


def test_run_python_is_a_throwaway_copy(tmp_path, monkeypatch):
    root, run = make_repo(tmp_path)
    monkeypatch.setattr(P, "TMP", tmp_path / "tmp")
    ws = P.Workspace.load(run)
    out = ws.run_python("import os\nprint(sorted(os.listdir('.')))\nopen('notes.md','w').write('x')\n")
    assert "exit 0" in out and "PROGRAMMER_BRIEF.md" in out and "passes" not in out
    assert not (run / "notes.md").exists()
    assert ws.run_python("s.measure('x')").startswith("refused")
    assert ws.run_python("import subprocess").startswith("refused")


class Scripted:
    """A fake provider that plays back tool calls, in the OpenAI message format."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []
        self.model = "fake-model"

    def start(self, system):
        return [{"role": "system", "content": system}]

    def add_user(self, conv, text):
        conv.append({"role": "user", "content": text})
        self.seen.append(text)

    def send(self, conv):
        calls = self.turns.pop(0)
        conv.append({"role": "assistant", "content": "", "n": len(calls)})
        return P.Turn("", [(f"c{i}", n, a) for i, (n, a) in enumerate(calls)],
                      {"input": 100, "output": 10, "reasoning": 5}, {"model": "fake-model-001"})

    def add_results(self, conv, results):
        for cid, _n, out in results:
            conv.append({"role": "tool", "tool_call_id": cid, "content": out})


def test_rounds_carry_the_conversation(tmp_path):
    root, run = make_repo(tmp_path)
    rel = run.relative_to(root)
    llm = Scripted([
        [("read_file", {"path": f"{rel}/PROGRAMMER_BRIEF.md"}), ("read_file", {"path": "docs/secret/REPORT.md"})],
        [("write_file", {"path": f"{rel}/program_01.py", "content": "print(1)\n"})],
        [("finish_round", {"summary": "program_01 written"})],
    ])
    r0 = P.run_round(run, "gemini", 0, llm=llm)
    assert r0["finished"] and r0["program"] == "program_01.py" and r0["steps"] == 3
    log = [json.loads(line) for line in (run / "programmer" / "transcript.jsonl").read_text().splitlines()]
    assert any(e.get("name") == "read_file" and e["result"].startswith("refused") for e in log)
    assert json.loads((run / "programmer" / "usage.json").read_text())["input"] == 300

    (run / "stdout_program_01.log").write_text("C0 0.4515\n")
    with pytest.raises(ValueError):
        P.run_round(run, "gemini", 2, run / "stdout_program_01.log", llm=Scripted([]))
    llm2 = Scripted([[("finish_round", {"summary": "nothing more"})]])
    r1 = P.run_round(run, "gemini", 1, run / "stdout_program_01.log", llm=llm2)
    assert r1["finished"] and "C0 0.4515" in llm2.seen[0]
    conv = json.loads((run / "programmer" / "conversation.json").read_text())
    assert conv[0]["role"] == "system" and sum(m["role"] == "user" for m in conv) == 2
    with pytest.raises(ValueError):
        P.run_round(run, "openai", 2, run / "stdout_program_01.log", llm=Scripted([]))


def test_round_without_finish_is_reported(tmp_path):
    root, run = make_repo(tmp_path)
    llm = Scripted([[], []])
    r = P.run_round(run, "gemini", 0, llm=llm)
    assert not r["finished"] and r["steps"] == 2


def test_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "KEY_DIR", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        P.api_key("gemini")
    f = tmp_path / "gemini.key"
    f.write_text("sekrit\n")
    os.chmod(f, 0o644)
    with pytest.raises(PermissionError):
        P.api_key("gemini")
    os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)
    assert P.api_key("gemini") == "sekrit"
    monkeypatch.setenv("GEMINI_API_KEY", "fromenv")
    assert P.api_key("gemini") == "fromenv"
    assert P.redact("url?key=fromenv", "fromenv") == "url?key=***"


def test_gemini_message_shapes():
    g = P.Gemini("m", "k", "https://x")
    conv = g.start("sys")
    g.add_user(conv, "hi")
    g.add_results(conv, [("call_0", "read_file", "text"), ("abc", "list_dir", "x")])
    parts = conv[-1]["parts"]
    assert parts[0]["functionResponse"] == {"name": "read_file", "response": {"result": "text"}}
    assert parts[1]["functionResponse"]["id"] == "abc"


def test_outside_section5():
    a = "1. **A**\n\n5. **Hypotheses**\n\n   x\n\n6. **C**\n"
    b = "1. **A**\n\n5. **Hypotheses**\n\n   y y y\n\n6. **C**\n"
    assert P.outside_section5(a) == P.outside_section5(b)
    assert P.outside_section5(a) != P.outside_section5(a.replace("**C**", "**D**"))


def _fake_server(reply):
    """A local HTTP server that records the request and answers with reply(path, body)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            out = json.dumps(reply(self.path, body)).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_openai_client_over_http():
    srv, seen = _fake_server(lambda path, body: {
        "id": "r1", "model": "gpt-x-2026", "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {
                "name": "read_file", "arguments": "{\"path\": \"wsp/ops.py\"}"}}]}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}}})
    try:
        c = P.OpenAIChat("gpt-x", "KEY", f"http://127.0.0.1:{srv.server_port}/v1")
        conv = c.start("sys"); c.add_user(conv, "go")
        t = c.send(conv)
        assert t.calls == [("t1", "read_file", {"path": "wsp/ops.py"})] and t.usage["reasoning"] == 2
        path, headers, body = seen[0]
        assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer KEY"
        assert {x["function"]["name"] for x in body["tools"]} == {t["name"] for t in P.TOOLS}
        c.add_results(conv, [("t1", "read_file", "x")])
        assert conv[-1] == {"role": "tool", "tool_call_id": "t1", "content": "x"}
    finally:
        srv.shutdown()


def test_gemini_client_over_http():
    srv, seen = _fake_server(lambda path, body: {
        "modelVersion": "gemini-4-argon-001", "responseId": "g1",
        "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
            {"text": "thinking", "thought": True},
            {"functionCall": {"name": "finish_round", "args": {"summary": "ok"}}, "thoughtSignature": "SIG"}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 4, "thoughtsTokenCount": 9}})
    try:
        g = P.Gemini("gemini-4-argon", "KEY", f"http://127.0.0.1:{srv.server_port}/v1beta")
        conv = g.start("sys"); g.add_user(conv, "go")
        t = g.send(conv)
        assert t.calls == [("call_1", "finish_round", {"summary": "ok"})] and t.text == ""
        assert t.usage == {"input": 11, "output": 4, "reasoning": 9}
        path, headers, body = seen[0]
        assert path == "/v1beta/models/gemini-4-argon:generateContent" and headers["x-goog-api-key"] == "KEY"
        assert body["systemInstruction"]["parts"][0]["text"] == "sys" and "_system" not in json.dumps(body["contents"])
        assert conv[-1]["parts"][1]["thoughtSignature"] == "SIG"  # sent back unchanged next turn
        g.send(conv)
        assert seen[1][2]["contents"][1]["parts"][1]["thoughtSignature"] == "SIG"
    finally:
        srv.shutdown()
