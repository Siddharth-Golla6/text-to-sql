"""Generate a short, grounded business-insight narrative for a query result."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from groq import Groq

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "narration.txt").read_text()


def narrate(question: str, sql: str, result: pd.DataFrame, client: Groq, model: str) -> str:
    if result is None or len(result) == 0:
        return "The query returned no rows. The filter may be too restrictive, or the data for the requested period/category may not exist in the database."

    preview = result.head(20)
    try:
        md = preview.to_markdown(index=False)
    except Exception:
        md = preview.to_string(index=False)

    prompt = PROMPT.format(question=question, sql=sql, result_md=md)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()
