"""io_utils.py — streaming JSONL parse (orjson if present, else stdlib json)."""

from pathlib import Path

try:
    import orjson

    def _loads(b):
        return orjson.loads(b)
except ImportError:  # pragma: no cover
    import json

    def _loads(b):
        return json.loads(b)


def stream_candidates(path):
    """Yield candidate dicts one at a time from a .jsonl file (memory-bounded)."""
    p = Path(path)
    with open(p, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except Exception:
                continue
