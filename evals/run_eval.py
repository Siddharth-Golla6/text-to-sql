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


def _row_as_str_tuple(row) -> tuple:
    """Row values as strings, handling NaN and dtype wobble."""
    out = []
    for v in row:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append("")
        elif isinstance(v, float):
            # Round and strip trailing zeros so 6.66 and 6.660000... compare equal.
            out.append(f"{round(v, 2):.2f}")
        else:
            out.append(str(v))
    return tuple(out)


def df_to_str_rows(df: pd.DataFrame) -> list[tuple]:
    """Convert a DataFrame to a list of string-tuples; ignores column names + dtypes."""
    if df is None:
        return []
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    return [_row_as_str_tuple(r) for r in d.itertuples(index=False, name=None)]


def _rows_match_with_ties(gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    """Order-sensitive compare that tolerates tied rows being in any order.

    Groups consecutive rows in gold whose sort-defining value (last column) is
    equal into a block; the predicted rows at those positions must be a
    permutation of the gold block.
    """
    if len(gold_rows) != len(pred_rows):
        return False
    # Find tie-blocks in gold using the LAST column (the one typically ordered by).
    # Fall back: no ties → straight compare.
    if gold_rows == pred_rows:
        return True
    i = 0
    while i < len(gold_rows):
        j = i + 1
        key = gold_rows[i][-1]
        while j < len(gold_rows) and gold_rows[j][-1] == key:
            j += 1
        if sorted(gold_rows[i:j]) != sorted(pred_rows[i:j]):
            return False
        i = j
    return True


def results_match(gold: pd.DataFrame, pred: pd.DataFrame, order_matters: bool) -> bool:
    g_rows = df_to_str_rows(gold)
    p_rows = df_to_str_rows(pred)
    if len(g_rows) != len(p_rows):
        return False
    if not g_rows:
        return True
    if len(g_rows[0]) != len(p_rows[0]):
        return False
    if not order_matters:
        return sorted(g_rows) == sorted(p_rows)
    return _rows_match_with_ties(g_rows, p_rows)


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

        passed = bool(passed)
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
