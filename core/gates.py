"""
core/gates.py

Declarative CI gates evaluated against a StageResult.summary. A gate is a dict::

    {"stage": "model", "metric": "r2", "op": ">=", "threshold": 0.5}

`metric` is a key looked up in the stage's summary dict (numeric). Gates are how
CI decides whether the pipeline may proceed (e.g. the model metric must pass
before `deploy` runs).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class GateResult:
    stage: str
    gate: str          # human label, e.g. "model.r2 >= 0.5"
    status: str        # pass | fail | skip
    observed: Any
    expected: Any
    message: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def blocking(self) -> bool:
        """A failed gate blocks the pipeline; a skipped one does not."""
        return self.status == "fail"


def _label(gate: dict[str, Any]) -> str:
    return f"{gate.get('stage')}.{gate.get('metric')} {gate.get('op')} {gate.get('threshold')}"


def evaluate_gate(stage: str, summary: dict[str, Any], gate: dict[str, Any]) -> GateResult:
    label = _label(gate)
    metric = gate.get("metric")
    op_name = gate.get("op", ">=")
    threshold = gate.get("threshold")
    op = _OPS.get(op_name)

    if op is None:
        return GateResult(stage, label, "skip", None, threshold,
                          f"unknown operator '{op_name}'")
    if metric not in (summary or {}):
        return GateResult(stage, label, "skip", None, threshold,
                          f"metric '{metric}' not in stage summary")

    observed = summary.get(metric)
    try:
        ok = op(float(observed), float(threshold))
    except (TypeError, ValueError):
        return GateResult(stage, label, "skip", observed, threshold,
                          f"metric '{metric}' is not numeric")

    status = "pass" if ok else "fail"
    msg = f"{metric}={observed} {op_name} {threshold} -> {status}"
    return GateResult(stage, label, status, observed, threshold, msg)


def gates_for_stage(gates: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [g for g in (gates or []) if g.get("stage") == stage]


def evaluate_gates(stage: str, summary: dict[str, Any],
                   gates: list[dict[str, Any]]) -> list[GateResult]:
    """Evaluate every gate declared for `stage` against its summary."""
    return [evaluate_gate(stage, summary, g) for g in gates_for_stage(gates, stage)]
