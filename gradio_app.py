"""Gradio front-end for the Text-to-SQL analytics assistant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.sql_agent import SQLAgent  # noqa: E402


EXAMPLES = [
    "How many customers do we have?",
    "Top 5 countries by revenue",
    "Monthly revenue trend in 2010",
    "Which artist earned the most revenue?",
    "Average invoice total per country",
    "Top 10 longest tracks in the catalog",
    "Which employee supports the most customers?",
    "How many tracks exist in each genre?",
]


ABOUT_MD = """
### Text-to-SQL Analytics Assistant

- Natural-language question in · Grounded SQL + chart + insight out
- **Dataset:** Chinook music store (DuckDB, 11 tables)
- **95% execution accuracy** on a 20-question benchmark

**Pipeline**
1. Embed question → retrieve top-k relevant tables (sentence-transformers)
2. LLM generates DuckDB SQL grounded on retrieved schema
3. `sqlglot` validator blocks DDL/DML and enforces `LIMIT`
4. DuckDB executes read-only
5. LLM writes a short grounded narrative on the result

**Model:** `qwen/qwen3.8-27b` on Groq free tier · [Source on GitHub](https://github.com/Siddharth-Golla6/text-to-sql)
"""


_agent: SQLAgent | None = None


def get_agent() -> SQLAgent:
    global _agent
    if _agent is None:
        _agent = SQLAgent()
    return _agent


def _pick_chart(df: pd.DataFrame):
    if df is None or df.empty or len(df.columns) < 2 or len(df) > 100:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]

    if date_cols and numeric_cols:
        return px.line(df, x=date_cols[0], y=numeric_cols[0], markers=True)
    if other_cols and numeric_cols:
        d = df.copy()
        if len(d) > 20:
            d = d.nlargest(20, numeric_cols[0])
        return px.bar(d, x=other_cols[0], y=numeric_cols[0])
    if len(numeric_cols) >= 2:
        return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
    return None


def _fmt_timing(ms: dict) -> str:
    if not ms:
        return "_(no timing)_"
    rows = [f"| {k.replace('_ms', '')} | {int(v)} ms |" for k, v in ms.items()]
    return "| step | ms |\n|---|---|\n" + "\n".join(rows)


def ask(question: str):
    """Full pipeline. Returns outputs in the order Gradio expects."""
    if not question or not question.strip():
        return (
            gr.update(value="Type a question first.", visible=True),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=pd.DataFrame(), visible=False),
            "", "", "",
        )

    try:
        agent = get_agent()
    except Exception as e:
        return (
            gr.update(value=f"**Agent failed to start:** {e}", visible=True),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=pd.DataFrame(), visible=False),
            "", "", "",
        )

    res = agent.ask(question, with_narrative=True)

    error_block = gr.update(visible=False)
    narrative_block = gr.update(visible=False)
    chart_block = gr.update(value=None, visible=False)
    rows_block = gr.update(value=pd.DataFrame(), visible=False)

    if res.error:
        error_block = gr.update(
            value=f"**Error:** {res.error}", visible=True
        )
    elif res.result is not None:
        if res.narrative:
            narrative_block = gr.update(
                value=f"**Insight.** {res.narrative}", visible=True
            )
        fig = _pick_chart(res.result)
        if fig is not None:
            chart_block = gr.update(value=fig, visible=True)
        rows_block = gr.update(
            value=res.result, visible=True,
            label=f"Rows returned: {len(res.result):,}",
        )

    sql_md = ""
    if res.sanitized_sql:
        sql_md = f"```sql\n{res.sanitized_sql}\n```"
    elif res.sql:
        sql_md = f"**Model SQL (rejected by validator):**\n```sql\n{res.sql}\n```"

    retrieved_md = ""
    if res.retrieved_tables:
        retrieved_md = ", ".join(f"`{t}`" for t in res.retrieved_tables)
    else:
        retrieved_md = "_(none)_"

    timing_md = _fmt_timing(res.timing_ms)

    return error_block, narrative_block, chart_block, rows_block, sql_md, retrieved_md, timing_md


with gr.Blocks(
    title="Text-to-SQL Analytics Assistant",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:
    gr.Markdown("# Text-to-SQL Analytics Assistant")
    gr.Markdown(
        "Ask a business question in English. Get a grounded SQL query, "
        "a chart, and a short insight — over an 11-table music-store "
        "warehouse."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. Which support rep brought in the most revenue?",
                lines=2,
                autofocus=True,
            )
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                clear_btn = gr.Button("Clear")

            gr.Markdown("**Try an example**")
            with gr.Row():
                example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLES[:4]]
            with gr.Row():
                example_btns += [gr.Button(ex, size="sm") for ex in EXAMPLES[4:]]

        with gr.Column(scale=2):
            gr.Markdown(ABOUT_MD)

    error_box = gr.Markdown(visible=False)
    narrative_box = gr.Markdown(visible=False)
    chart_box = gr.Plot(visible=False, label="Chart")
    rows_box = gr.Dataframe(visible=False, label="Rows", wrap=True)

    with gr.Accordion("Generated SQL", open=False):
        sql_box = gr.Markdown()
    with gr.Accordion("Retrieved tables (top-k)", open=False):
        retrieved_box = gr.Markdown()
    with gr.Accordion("Timing", open=False):
        timing_box = gr.Markdown()

    outputs = [error_box, narrative_box, chart_box, rows_box, sql_box, retrieved_box, timing_box]

    run_btn.click(ask, inputs=question, outputs=outputs)
    question.submit(ask, inputs=question, outputs=outputs)

    for btn, ex in zip(example_btns, EXAMPLES):
        btn.click(lambda e=ex: e, outputs=question).then(
            ask, inputs=question, outputs=outputs
        )

    def _clear():
        return (
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=pd.DataFrame(), visible=False),
            "", "", "",
        )

    clear_btn.click(_clear, outputs=[question, *outputs])
