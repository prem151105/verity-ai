"""
Shared audit logger.
Writes structured JSONL trace entries for every agent node execution.
Enables full replay and compliance-style review of any research run.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Append-only JSONL audit log per research run.
    Each entry captures a full node execution record.
    """

    def __init__(self, audit_log_dir: str, run_id: str):
        Path(audit_log_dir).mkdir(parents=True, exist_ok=True)
        self._log_path = Path(audit_log_dir) / f"{run_id}.jsonl"
        self._run_id = run_id

    def log(
        self,
        node: str,
        inputs: dict,
        outputs: dict,
        tool_calls: list[dict] | None = None,
        duration_seconds: float = 0.0,
    ) -> dict:
        """
        Write a trace entry to the JSONL log.

        Returns:
            The serialized trace entry dict (also appended to state.trace).
        """
        entry = {
            "run_id": self._run_id,
            "node": node,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 3),
            "inputs": _sanitize(inputs),
            "outputs": _sanitize(outputs),
            "tool_calls": tool_calls or [],
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    @property
    def log_path(self) -> str:
        return str(self._log_path)


def _sanitize(obj: object, max_str_len: int = 2000) -> object:
    """
    Recursively sanitize an object for JSON serialization.
    Truncates very long strings to keep logs readable.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v, max_str_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v, max_str_len) for v in obj[:50]]  # cap lists at 50 items
    if isinstance(obj, str):
        return obj[:max_str_len] + "…" if len(obj) > max_str_len else obj
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    return str(obj)
