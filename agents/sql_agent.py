"""
End-to-end SQL agent.

Pipeline:
    user question
      -> schema retriever (top-k tables)
      -> Groq LLM generates SQL
      -> safety validator (sqlglot) sanitizes + adds LIMIT
      -> DuckDB executor (read-only, timeout)
      -> narrator (short grounded insight)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from .narrator import narrate
from .safety_validator import validate_sql
from .schema_retriever import SchemaRetriever, TableProfile

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "sql_generation.txt"
CACHE_PATH = PROJECT_ROOT / "data" / "schema_cache.json"


@dataclass
class AgentResult:
    question: str
    sql: str
    sanitized_sql: str
    result: pd.DataFrame | None
    narrative: str
    retrieved_tables: list[str] = field(default_factory=list)
    error: str = ""
    timing_ms: dict[str, float] = field(default_factory=dict)


class SQLAgent:
    def __init__(
        self,
        db_path: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ):
        self.db_path = db_path or os.getenv("DB_PATH", "data/chinook.duckdb")
        if not Path(self.db_path).is_absolute():
            self.db_path = str(PROJECT_ROOT / self.db_path)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.top_k = int(os.getenv("TOP_K_TABLES", str(top_k)))

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = Groq(api_key=api_key)

        self.retriever = SchemaRetriever(self.db_path, top_k=self.top_k)
        if not self.retriever.load_cache(CACHE_PATH):
            self.retriever.build()
            self.retriever.save_cache(CACHE_PATH)

        self._prompt_template = PROMPT_PATH.read_text()

    def _generate_sql(self, question: str, tables: list[TableProfile]) -> str:
        schema_block = "\n\n".join(t.to_prompt_block() for t in tables)
        prompt = self._prompt_template.format(schema=schema_block, question=question)
        # Groq's gpt-oss-* models reserve tokens for a hidden reasoning channel,
        # which can truncate the SQL. 2000 leaves plenty of headroom for both.
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 2000,
        }
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = "low"

        # Retry on free-tier rate limits (Groq caps at 8000 TPM).
        from groq import RateLimitError
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    def _execute(self, sql: str) -> pd.DataFrame:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def ask(self, question: str, with_narrative: bool = True) -> AgentResult:
        timing: dict[str, float] = {}
        retrieved: list[TableProfile] = []
        sql = ""
        sanitized = ""
        df: pd.DataFrame | None = None
        narrative = ""
        error = ""

        try:
            t0 = time.time()
            retrieved = self.retriever.retrieve(question)
            timing["retrieve_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            sql = self._generate_sql(question, retrieved)
            timing["generate_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            val = validate_sql(sql)
            timing["validate_ms"] = (time.time() - t0) * 1000
            if not val.ok:
                error = f"Safety validator rejected the query: {val.reason}"
                return AgentResult(
                    question=question,
                    sql=sql,
                    sanitized_sql="",
                    result=None,
                    narrative="",
                    retrieved_tables=[t.name for t in retrieved],
                    error=error,
                    timing_ms=timing,
                )
            sanitized = val.sql

            t0 = time.time()
            df = self._execute(sanitized)
            timing["execute_ms"] = (time.time() - t0) * 1000

            if with_narrative:
                t0 = time.time()
                narrative = narrate(question, sanitized, df, self.client, self.model)
                timing["narrate_ms"] = (time.time() - t0) * 1000

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        return AgentResult(
            question=question,
            sql=sql,
            sanitized_sql=sanitized,
            result=df,
            narrative=narrative,
            retrieved_tables=[t.name for t in retrieved],
            error=error,
            timing_ms=timing,
        )
