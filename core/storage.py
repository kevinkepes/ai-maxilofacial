"""Istoricul analizelor si feedback-ul medicului (SQLite local)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/analize.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS analize (
               id TEXT PRIMARY KEY,
               patient_code TEXT,
               filename TEXT,
               created_at TEXT,
               model TEXT,
               n_findings INTEGER,
               report_json TEXT)"""
    )
    return con


def save(report: dict) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO analize VALUES (?,?,?,?,?,?,?)",
            (
                report["analysis_id"],
                report["patient_code"],
                report["filename"],
                report["created_at"],
                report["model"],
                len(report["findings"]),
                json.dumps(report, ensure_ascii=False),
            ),
        )


def history(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, patient_code, filename, created_at, model, n_findings "
            "FROM analize ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    cols = ["ID", "Pacient", "Fisier", "Data", "Model", "Leziuni"]
    return [dict(zip(cols, r)) for r in rows]
