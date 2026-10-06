"""SQLite snapshots for one local API service; terminal runs survive restarts."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from config import settings


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@contextmanager
def connect():
    root = Path(settings.audit_log_dir)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'runs.sqlite3', timeout=15)
    try:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            yield db
    finally:
        db.close()


def save(run):
    with connect() as db:
        db.execute('INSERT OR REPLACE INTO runs VALUES (?, ?)',
                   (run['run_id'], json.dumps(_json_safe(run), default=str)))


def load(run_id):
    with connect() as db:
        row = db.execute('SELECT payload FROM runs WHERE id = ?', (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


def recent():
    with connect() as db:
        rows = db.execute('SELECT payload FROM runs ORDER BY rowid DESC LIMIT 100').fetchall()
    return [json.loads(row[0]) for row in rows]
