"""
Execution-accuracy eval harness.

For each (question, gold_sql) in questions.jsonl:
  - Run the agent to get model_sql
  - Execute both gold_sql and model_sql against DuckDB
  - Compare the two result sets using a semantic comparator
    (column-order-independent, row-order-independent for aggregates,
    row-order-preserving when the question asks for ordering).
  - Record pass/fail, SQL validity, latency, and error.

Usage:
    python evals/run_eval.py
    python evals/run_eval.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, median

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.sql_agent import SQLAgent  # noqa: E402

QUESTIONS = PROJECT_ROOT / "evals" / "questions.jsonl"
REPORT_DIR = PROJECT_ROOT / "evals"


def normalize(df: pd.DataFrame, order_matters: bool) -> pd.DataFrame:
    """Normalize a result for comparison."""
    if df is None:
        return pd.DataFrame()
    d = df.copy()
    # Round all floats to 2 decimals so Groq's casting choices don't trip us up.
    for c in d.select_dtypes(include="float").columns:
        d[c] = d[c].round(2)
    # Convert dates to strings.
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    # Sort rows if order doesn't matter.
    if not order_matters:
        d = d.sort_values(by=list(d.columns)).reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)
    return d


def results_match(gold: pd.DataFrame, pred: pd.DataFrame, order_matters: bool) -> bool:
    g = normalize(gold, order_matters)
    p = normalize(pred, order_matters)

    # Shape must match.
    if g.shape != p.shape:
        return False

    # Compare values set-wise per row (column-order-independent).
    g_rows = sorted(tuple(sorted(zip(g.columns, row))) for row in g.itertuples(index=False, name=None))
    p_rows = sorted(tuple(sorted(zip(p.columns, row))) for row in p.itertuples(index=False, name=None))

    if g_rows == p_rows:
        return True

    # Fall back to position-based numeric comparison (handles different aliasing).
    try:
        return (g.values == p.values).all()
    except Exception:
        return False


ORDER_KEYWORDS = re.compile(r"\b(top|first|last|highest|lowest|most|least|order|rank|trend|over time|monthly|yearly|by month|by year)\b", re.I)


def question_requires_order(q: str) -> bool:
    return bool(ORDER_KEYWORDS.search(q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    args = parser.parse_args()

    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    agent = SQLAgent()
    db_path = agent.db_path

    results = []
    passes = 0
    valid_sql = 0

    for i, row in enumerate(rows, 1):
        qid, q, gold_sql = row["id"], row["question"], row["gold_sql"]
        print(f"[{i}/{len(rows)}] Q{qid}: {q}")

        order_matters = question_requires_order(q)

        t0 = time.time()
        res = agent.ask(q, with_narrative=False)
        total_ms = (time.time() - t0) * 1000

        passed = False
        reason = ""
        pred_df = None

        if res.error:
            reason = res.error
        else:
            valid_sql += 1
            try:
                con = duckdb.connect(db_path, read_only=True)
                gold_df = con.execute(gold_sql).fetchdf()
                pred_df = res.result
                con.close()
                passed = results_match(gold_df, pred_df, order_matters)
                if not passed:
                    reason = f"result mismatch (gold shape {gold_df.shape} vs pred shape {pred_df.shape if pred_df is not None else None})"
            except Exception as e:
                reason = f"gold_sql failed: {e}"

        passes += int(passed)

        results.append({
            "id": qid,
            "difficulty": row.get("difficulty", ""),
            "question": q,
            "passed": passed,
            "reason": reason,
            "model_sql": res.sanitized_sql or res.sql,
            "gold_sql": gold_sql,
            "total_ms": round(total_ms, 1),
            "retrieve_ms": round(res.timing_ms.get("retrieve_ms", 0), 1),
            "generate_ms": round(res.timing_ms.get("generate_ms", 0), 1),
            "execute_ms": round(res.timing_ms.get("execute_ms", 0), 1),
        })
        print(f"    -> {'PASS' if passed else 'FAIL'} ({total_ms:.0f} ms){' - ' + reason if reason else ''}")

    # Aggregate
    n = len(results)
    exec_acc = passes / n if n else 0
    sql_valid = valid_sql / n if n else 0
    latencies = [r["total_ms"] for r in results]

    summary = {
        "n_questions": n,
        "execution_accuracy": round(exec_acc, 3),
        "sql_validity_rate": round(sql_valid, 3),
        "latency_mean_ms": round(mean(latencies), 1) if latencies else 0,
        "latency_median_ms": round(median(latencies), 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1) if latencies else 0,
    }

    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Breakdown by difficulty
    by_diff: dict[str, list] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["passed"])
    print("\n  Accuracy by difficulty:")
    for d, lst in by_diff.items():
        print(f"    {d}: {sum(lst)}/{len(lst)}  ({sum(lst) / len(lst):.0%})")

    (REPORT_DIR / "last_run.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, default=str)
    )
    # Also write a readable Markdown report
    md = ["# Eval Report", ""]
    md.append("## Summary")
    for k, v in summary.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## By difficulty")
    for d, lst in by_diff.items():
        md.append(f"- **{d}**: {sum(lst)}/{len(lst)} ({sum(lst) / len(lst):.0%})")
    md.append("")
    md.append("## Per-question results")
    md.append("")
    md.append("| id | difficulty | passed | total_ms | question |")
    md.append("|---|---|---|---|---|")
    for r in results:
        md.append(f"| {r['id']} | {r['difficulty']} | {'OK' if r['passed'] else 'FAIL'} | {r['total_ms']} | {r['question']} |")
    (REPORT_DIR / "last_run.md").write_text("\n".join(md))
    print(f"\n  Saved: {REPORT_DIR / 'last_run.json'}")
    print(f"  Saved: {REPORT_DIR / 'last_run.md'}")


if __name__ == "__main__":
    main()
