"""Structured JSON logging. Owner: Person 3."""

import json
import uuid
from datetime import UTC, datetime

from src.settings import RUN_LOG_PATH

# Keys that could carry document text. They are never written to the log.
_BLOCKED_KEYS = {"text", "content", "page_content", "document", "documents"}
_MAX_STR_LEN = 300  # long strings are truncated so a full passage cannot slip in


def new_run_id() -> str:
    """Return a fresh UUID string for one question."""
    return str(uuid.uuid4())


def _clean(value):
    """Make a value JSON-safe and short; recurse into dicts and lists."""
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR_LEN else value[:_MAX_STR_LEN] + "..."
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if k not in _BLOCKED_KEYS}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def chunk_metadata(chunks) -> list[dict]:
    """Reduce retrieved chunks to what is safe to log: file, chunk id, score."""
    return [
        {
            "source_filename": c.source_filename,
            "chunk_id": c.chunk_id,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]


def log_event(run_id: str, event: str, **data) -> None:
    """Append one JSON line to RUN_LOG_PATH. Log filenames/chunk IDs, not full text."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "run_id": run_id,
        "event": event,
        **_clean(data),
    }
    try:
        RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never crash a user request
