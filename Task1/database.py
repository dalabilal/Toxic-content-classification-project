"""SQLite database: one row for every text / image the user submits."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "toxic_logs.db"


def connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = connect()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS logs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               created_at TEXT NOT NULL,
               input_type TEXT NOT NULL,
               content TEXT NOT NULL,
               predicted_label TEXT NOT NULL,
               confidence REAL NOT NULL
           )"""
    )
    conn.commit()
    conn.close()


def log_entry(input_type, content, predicted_label, confidence):
    """Save one submission. For images, content is the generated caption."""
    conn = connect()
    conn.execute(
        "INSERT INTO logs (created_at, input_type, content, predicted_label, confidence) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), input_type, content, predicted_label, float(confidence)),
    )
    conn.commit()
    conn.close()


def get_logs(input_type=None):
    """Read all rows (newest first). input_type can be 'text', 'image' or None for everything."""
    query = "SELECT * FROM logs"
    params = ()
    if input_type:
        query += " WHERE input_type = ?"
        params = (input_type,)
    query += " ORDER BY id DESC"
    conn = connect()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df