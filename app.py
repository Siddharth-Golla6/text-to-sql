"""HF Spaces entry point (Gradio).

Bootstraps the DuckDB from the Chinook SQLite source on first run (the
database file is kept out of the repo via .gitignore), then launches the
Gradio UI defined in `app/gradio_app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "chinook.duckdb"
if not DB_PATH.exists():
    from data import setup_db
    print("[boot] First-run setup: downloading Chinook and building DuckDB...")
    setup_db.download_chinook()
    setup_db.build_duckdb()
    print("[boot] DuckDB ready.")

from gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
