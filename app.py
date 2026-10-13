"""HF Spaces entry point.

On the Space container the database file is not shipped in the repo (the
.gitignore keeps `.duckdb` out, and SQLite + DuckDB together are small enough
that building them on startup is cheaper than committing them). On first run
this file bootstraps the DuckDB from the Chinook SQLite source, then hands off
to the real Streamlit app.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "chinook.duckdb"
if not DB_PATH.exists():
    import streamlit as st
    with st.spinner("First-run setup: downloading Chinook and building DuckDB (one-time, ~10s)..."):
        from data import setup_db
        setup_db.download_chinook()
        setup_db.build_duckdb()

runpy.run_path(str(ROOT / "app" / "streamlit_app.py"), run_name="__main__")
