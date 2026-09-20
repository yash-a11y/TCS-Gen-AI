"""SQLite helpers for structured customer and ticket data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import DB_PATH, ensure_data_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    plan TEXT NOT NULL,
    status TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    account_manager TEXT
);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_data_dirs()
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    own_connection = conn is None
    conn = conn or get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        if own_connection:
            conn.close()


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def get_customer_profile(name: str) -> dict | None:
    """Find one customer by name (partial match, case-insensitive)."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, name, email, phone, plan, status, signup_date, account_manager
            FROM customers
            WHERE name LIKE ?
            COLLATE NOCASE
            ORDER BY id
            LIMIT 1
            """,
            (f"%{name.strip()}%",),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_customer_tickets(name: str) -> list[dict]:
    """Tickets for the customer matched by name."""
    profile = get_customer_profile(name)
    if profile is None:
        return []
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, customer_id, subject, description, status, priority, created_at, resolved_at
            FROM tickets
            WHERE customer_id = ?
            ORDER BY created_at DESC
            """,
            (profile["id"],),
        ).fetchall()
        return rows_to_dicts(rows)
    finally:
        conn.close()


if __name__ == "__main__":
    profile = get_customer_profile("Ema")
    tickets = get_customer_tickets("Ema")
    print("profile:", profile)
    print("ticket_count:", len(tickets))
    for ticket in tickets:
        print(f"  #{ticket['id']} [{ticket['status']}] {ticket['subject']}")

