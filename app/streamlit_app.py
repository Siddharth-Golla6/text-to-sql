"""Streamlit front-end for the Text-to-SQL analytics assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.sql_agent import SQLAgent  # noqa: E402

st.set_page_config(
    page_title="Text-to-SQL Analytics Assistant",
    layout="wide",
    page_icon="SQL",
)


# ------------------------------- helpers -------------------------------
def pick_chart(df: pd.DataFrame):
    """Heuristic chart-type picker. Returns a Plotly figure or None."""
    if df is None or df.empty or len(df.columns) < 2:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]

    if len(df) > 100:
        return None  # too many rows to chart meaningfully

    # Time series
    if date_cols and numeric_cols:
        return px.line(df, x=date_cols[0], y=numeric_cols[0])

    # Categorical bar
    if other_cols and numeric_cols:
        d = df.copy()
        if len(d) > 20:
            d = d.nlargest(20, numeric_cols[0])
        return px.bar(d, x=other_cols[0], y=numeric_cols[0])

    # Scatter
    if len(numeric_cols) >= 2:
        return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])

    return None


@st.cache_resource(show_spinner="Warming up the schema retriever...")
def get_agent() -> SQLAgent:
    return SQLAgent()


# ------------------------------- sidebar -------------------------------
with st.sidebar:
    st.title("About")
    st.markdown(
        "**Text-to-SQL Analytics Assistant**\n\n"
        "- Natural-language question in\n"
        "- Grounded SQL + chart + insight out\n\n"
        "**Database:** Chinook music store (DuckDB)\n\n"
        "**Pipeline**\n"
        "1. Embed the question and retrieve top-k relevant tables\n"
        "2. LLM generates DuckDB SQL with few-shot grounding\n"
        "3. `sqlglot` validator blocks DDL/DML, enforces `LIMIT`\n"
        "4. DuckDB executes read-only\n"
        "5. LLM writes a short grounded narrative"
    )
    st.markdown("---")
    st.caption("Example questions")
    examples = [
        "How many customers do we have?",
        "Top 5 countries by revenue",
        "Monthly revenue trend in 2010",
        "Which artist earned the most revenue?",
        "Average invoice total per country",
        "Top 10 longest tracks in the catalog",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["q_input"] = ex


# ------------------------------- main -----------------------------------
st.title("Text-to-SQL Analytics Assistant")

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Agent could not start: {e}")
    st.stop()

q = st.text_input(
    "Ask a question about the music store data",
    key="q_input",
    placeholder="e.g. Which support rep brought in the most revenue?",
)

run = st.button("Run", type="primary")

if run and q:
    with st.spinner("Thinking..."):
        res = agent.ask(q)

    if res.error:
        st.error(res.error)
        if res.sql:
            with st.expander("Model SQL (rejected)"):
                st.code(res.sql, language="sql")

    if res.result is not None and not res.error:
        if res.narrative:
            st.success(res.narrative)

        fig = pick_chart(res.result)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Rows returned:** {len(res.result):,}")
        st.dataframe(res.result, use_container_width=True)

    if res.sanitized_sql or res.sql:
        with st.expander("Show SQL"):
            if res.sanitized_sql:
                st.code(res.sanitized_sql, language="sql")
            if res.sql and res.sql.strip() != res.sanitized_sql.strip():
                st.caption("Original model output")
                st.code(res.sql, language="sql")

    with st.expander("Retrieved tables (top-k)"):
        st.write(res.retrieved_tables or "none")

    with st.expander("Timing (ms)"):
        st.json(res.timing_ms)

elif run and not q:
    st.warning("Type a question first.")

st.markdown("---")
st.caption(
    "Built as a portfolio project. Source: https://github.com/<your-user>/text-to-sql"
)
