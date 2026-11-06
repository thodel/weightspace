"""A session: hypotheses first, edits logged, passes budgeted, one freeze."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from wsp import diff as diffmod
from wsp.metrics import Metrics, score, per_line_edits, bootstrap_delta
from wsp.subject import Subject


class BudgetExhausted(RuntimeError):
    pass


class SessionClosed(RuntimeError):
    pass


@dataclass
class Session:
    subject: Subject
    practice: Sequence[dict]            # items with "key" and "text" (reference)
    run_dir: Path
    budget: int = 25
    seed: int = 20261106
    practice_id: str = ""               # sha256 of the line list, from preregistration
    _log: list[dict] = field(default_factory=list, init=False)
    _used: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _start_tensors: dict = field(default_factory=dict, init=False)
    _last_edit: dict | None = field(default=None, init=False)
    _pre_edit_tensors: dict | None = field(default=None, init=False)
    _baseline: Metrics | None = field(default=None, init=False)
    _last_lines: list | None = field(default=None, init=False)
    _baseline_lines: list | None = field(default=None, init=False)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_tensors = {k: v.copy() for k, v in self.subject.tensors().items()}
        counter = self.run_dir / "budget.json"
        if counter.exists():
            self._used = json.loads(counter.read_text())["used"]
        self._event("start", budget=self.budget, used=self._used, digest=self.subject.digest(),
                    practice_lines=len(self.practice), practice_id=self.practice_id,
                    describe=self.subject.describe())

    # -- logging --------------------------------------------------------
    def _event(self, kind: str, **fields) -> dict:
        if self._closed:
            raise SessionClosed("session is frozen")
        e = {"t": time.time(), "kind": kind, "budget_used": self._used, **fields}
        self._log.append(e)
        with (self.run_dir / "log.jsonl").open("a") as fh:
            fh.write(json.dumps(e, default=_jsonable) + "\n")
        return e

    @property
    def budget_left(self) -> int:
        return self.budget - self._used

    # -- the protocol ---------------------------------------------------
    def hypothesis(self, text: str, prediction: str, edit_class: str | None = None) -> int:
        e = self._event("hypothesis", text=text, prediction=prediction, edit_class=edit_class)
        return len(self._log) - 1

    def note(self, text: str) -> None:
        self._event("note", text=text)

    def edit(self, op: str, names: Sequence[str], hypothesis: int | None = None, **kwargs) -> int:
        if self._last_edit is not None:
            raise RuntimeError("decide on the previous edit (keep/revert) first")
        self._pre_edit_tensors = {n: self.subject.tensors()[n].copy() for n in names}
        self.subject.apply(op, names, **kwargs)
        self._last_edit = self._event("edit", op=op, names=list(names), hypothesis=hypothesis,
                                      args={k: _summ(v) for k, v in kwargs.items()},
                                      digest=self.subject.digest())
        return len(self._log) - 1

    def measure(self, label: str = "") -> Metrics:
        if self._used >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} passes used")
        self._used += 1
        (self.run_dir / "budget.json").write_text(json.dumps({"used": self._used, "budget": self.budget}))
        preds = self.subject.forward(self.practice)
        hyps = [p.text for p in preds]
        refs = [item["text"] for item in self.practice]
        m = score(hyps, refs)
        lines = per_line_edits(hyps, refs)
        readings = self.run_dir / "readings"
        readings.mkdir(exist_ok=True)
        (readings / f"pass_{self._used:03d}.jsonl").write_text(
            "".join(json.dumps({"key": p.key, "text": p.text, "extra": p.extra}) + "\n" for p in preds))
        if self._baseline is None:
            self._baseline, self._baseline_lines = m, lines
        self._last_lines = lines
        self._event("pass", label=label, metrics=m.as_dict(), digest=self.subject.digest(),
                    vs_baseline=bootstrap_delta(self._baseline_lines, lines, resamples=500, seed=self.seed)
                    if self._baseline_lines is not None else None)
        return m

    def keep(self, reason: str = "") -> None:
        self._decide(True, reason)

    def revert(self, reason: str = "") -> None:
        if self._pre_edit_tensors:
            for n, v in self._pre_edit_tensors.items():
                self.subject.set_tensor(n, v)
        self._decide(False, reason)

    def _decide(self, kept: bool, reason: str) -> None:
        if self._last_edit is None:
            raise RuntimeError("no edit to decide on")
        self._event("decision", kept=kept, reason=reason, edit=self._last_edit.get("op"),
                    digest=self.subject.digest())
        self._last_edit = None
        self._pre_edit_tensors = None

    def freeze(self) -> dict:
        if self._last_edit is not None:
            raise RuntimeError("decide on the last edit before freezing")
        sparse = diffmod.sparse_diff(self._start_tensors, self.subject.tensors())
        diffmod.save_sparse_diff(sparse, self.run_dir / "diff.json")
        cmp = diffmod.compare(self._start_tensors, self.subject.tensors())
        summary = {
            "budget": self.budget, "used": self._used, "seed": self.seed,
            "practice_lines": len(self.practice), "practice_id": self.practice_id,
            "start_digest": self._log[0]["digest"], "final_digest": self.subject.digest(),
            "edits": sum(1 for e in self._log if e["kind"] == "edit"),
            "kept": sum(1 for e in self._log if e["kind"] == "decision" and e["kept"]),
            "baseline": self._baseline.as_dict() if self._baseline else None,
            "last_pass": next((e["metrics"] for e in reversed(self._log) if e["kind"] == "pass"), None),
            "changed_tensors": {k: v for k, v in cmp.items() if v.get("changed")},
            "heldout": None, "deviations": [],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
        self._event("freeze", summary_written=True)
        self._closed = True
        return summary


def _summ(v):
    if isinstance(v, np.ndarray):
        return {"array": list(v.shape), "norm": float(np.linalg.norm(v))}
    return v


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
