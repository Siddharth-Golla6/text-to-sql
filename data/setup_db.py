"""
Download the Chinook SQLite database and load it into a local DuckDB file.

Chinook is a well-known sample schema (music store) with 11 tables, foreign keys,
and realistic transactional data. We mirror it into DuckDB so the app runs on
a single zero-dependency analytical engine.

Run:  python data/setup_db.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import requests

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SQLITE_PATH = HERE / "chinook.sqlite"
DUCKDB_PATH = HERE / "chinook.duckdb"

CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)


def download_chinook() -> None:
    if SQLITE_PATH.exists() and SQLITE_PATH.stat().st_size > 500_000:
        print(f"[setup] Chinook SQLite already present at {SQLITE_PATH}")
        return
    print(f"[setup] Downloading Chinook from {CHINOOK_URL} ...")
    r = requests.get(CHINOOK_URL, timeout=60)
    r.raise_for_status()
    SQLITE_PATH.write_bytes(r.content)
    print(f"[setup] Saved {len(r.content):,} bytes to {SQLITE_PATH}")


def build_duckdb() -> None:
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    print(f"[setup] Building DuckDB at {DUCKDB_PATH} ...")

    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{SQLITE_PATH.as_posix()}' AS src (TYPE sqlite);")

    tables = [row[0] for row in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = 'src' AND table_name NOT LIKE 'sqlite_%' "
        "ORDER BY table_name"
    ).fetchall()]

    for t in tables:
        con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM src."{t}";')
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  - {t}: {n:,} rows")

    con.execute("DETACH src;")
    con.close()
    print(f"[setup] Done. {len(tables)} tables copied into DuckDB.")


def smoke_test() -> None:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    tables = con.execute("SHOW TABLES").fetchdf()
    print("\n[smoke-test] Tables in chinook.duckdb:")
    print(tables.to_string(index=False))
    sample = con.execute(
        "SELECT c.Country, COUNT(*) AS invoices, ROUND(SUM(i.Total), 2) AS revenue "
        "FROM Invoice i JOIN Customer c USING(CustomerId) "
        "GROUP BY c.Country ORDER BY revenue DESC LIMIT 5"
    ).fetchdf()
    print("\n[smoke-test] Top 5 countries by revenue:")
    print(sample.to_string(index=False))
    con.close()


if __name__ == "__main__":
    try:
        download_chinook()
        build_duckdb()
        smoke_test()
    except Exception as e:
        print(f"[setup] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
