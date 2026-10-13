"""
Schema-aware table retriever.

Builds a natural-language profile for every table in the DuckDB (columns, types,
primary and foreign keys, sample values, row counts) and embeds each profile
with sentence-transformers. At query time, the top-k most relevant tables are
returned so the SQL generator sees a small, focused schema instead of the full
warehouse (which breaks when the schema grows to hundreds of tables).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TableProfile:
    name: str
    description: str
    row_count: int
    columns: list
    foreign_keys: list

    def to_prompt_block(self) -> str:
        cols = "\n".join(
            f"    - {c['name']} ({c['type']})"
            + (f" sample={c['sample']}" if c.get("sample") else "")
            for c in self.columns
        )
        fks = ""
        if self.foreign_keys:
            fk_lines = "\n".join(
                f"    - {fk['from']} -> {fk['to_table']}.{fk['to_column']}"
                for fk in self.foreign_keys
            )
            fks = f"\n  Foreign keys:\n{fk_lines}"
        return (
            f'TABLE "{self.name}" ({self.row_count:,} rows)\n'
            f"  Description: {self.description}\n"
            f"  Columns:\n{cols}{fks}"
        )


class SchemaRetriever:
    def __init__(self, db_path: str, top_k: int = 5):
        self.db_path = db_path
        self.top_k = top_k
        self._profiles: list[TableProfile] = []
        self._embeddings: np.ndarray | None = None
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def build(self) -> None:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            for t in tables:
                self._profiles.append(self._profile_table(con, t))
        finally:
            con.close()
        docs = [p.description + "\n" + p.to_prompt_block() for p in self._profiles]
        self._embeddings = self.model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )

    def _profile_table(self, con: duckdb.DuckDBPyConnection, table: str) -> TableProfile:
        schema_df = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
        row_count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        columns = []
        for _, row in schema_df.iterrows():
            col = {"name": row["name"], "type": row["type"]}
            try:
                sample = con.execute(
                    f'SELECT "{row["name"]}" FROM "{table}" '
                    f'WHERE "{row["name"]}" IS NOT NULL LIMIT 1'
                ).fetchone()
                if sample is not None and sample[0] is not None:
                    v = str(sample[0])
                    col["sample"] = v[:40] + ("..." if len(v) > 40 else "")
            except Exception:
                pass
            columns.append(col)

        fks = self._foreign_keys(con, table)
        description = self._table_description(table, [c["name"] for c in columns])
        return TableProfile(
            name=table,
            description=description,
            row_count=row_count,
            columns=columns,
            foreign_keys=fks,
        )

    @staticmethod
    def _foreign_keys(con: duckdb.DuckDBPyConnection, table: str) -> list:
        # DuckDB doesn't preserve FKs on CTAS from SQLite, so infer them:
        # any column named Xxx(Id) that matches a table named Xxx is an FK.
        all_tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        # PRAGMA table_info rows are (cid, name, type, notnull, dflt_value, pk).
        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        fks = []
        for c in cols:
            if c.endswith("Id") and c != table + "Id":
                ref = c[:-2]
                if ref in all_tables and c != "ReportsTo":
                    fks.append({"from": c, "to_table": ref, "to_column": c})
        if table == "Employee" and "ReportsTo" in cols:
            fks.append({"from": "ReportsTo", "to_table": "Employee", "to_column": "EmployeeId"})
        return fks

    @staticmethod
    def _table_description(table: str, columns: Sequence[str]) -> str:
        """Short human-readable description to help semantic matching."""
        hints = {
            "Album": "Music album catalog. Each album belongs to an artist and contains tracks.",
            "Artist": "Musicians / bands who recorded albums.",
            "Customer": "End customers who purchase music. Contains geography and support rep.",
            "Employee": "Staff of the music store. Includes managers and sales support reps.",
            "Genre": "Music genres such as Rock, Jazz, Classical.",
            "Invoice": "Sales invoice header. One row per purchase order with total, date, billing country.",
            "InvoiceLine": "Line items of invoices. Links invoice to track and quantity sold.",
            "MediaType": "Media formats such as MPEG audio, AAC audio, Protected AAC audio.",
            "Playlist": "User-defined collections of tracks.",
            "PlaylistTrack": "Many-to-many between playlists and tracks.",
            "Track": "Individual songs. Links to album, genre, media type; has duration, size, price.",
        }
        return hints.get(table, f"Table {table} with columns: {', '.join(columns)}")

    def retrieve(self, question: str) -> List[TableProfile]:
        if self._embeddings is None:
            raise RuntimeError("Call build() first")
        q_emb = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self._embeddings @ q_emb
        idx = np.argsort(-scores)[: self.top_k]
        return [self._profiles[i] for i in idx]

    def all_profiles(self) -> List[TableProfile]:
        return list(self._profiles)

    def save_cache(self, path: str | Path) -> None:
        data = {
            "profiles": [
                {
                    "name": p.name,
                    "description": p.description,
                    "row_count": p.row_count,
                    "columns": p.columns,
                    "foreign_keys": p.foreign_keys,
                }
                for p in self._profiles
            ],
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else None,
        }
        Path(path).write_text(json.dumps(data))

    def load_cache(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        self._profiles = [TableProfile(**row) for row in data["profiles"]]
        emb = data.get("embeddings")
        self._embeddings = np.array(emb) if emb else None
        return True
