"""
SQL safety validator.

Parses generated SQL with sqlglot (an actual SQL parser, not regex) and rejects
anything that isn't a plain SELECT / WITH. Also normalizes the query: adds a
LIMIT if the user forgot one, and strips trailing semicolons / Markdown fences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp

DEFAULT_LIMIT = 500

FORBIDDEN = {
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,  # catches ATTACH, PRAGMA, etc. in some dialects
}


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    reason: str = ""


def _strip_fences(sql: str) -> str:
    s = sql.strip()
    m = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    return s.rstrip(";").strip()


def validate_sql(sql: str, default_limit: int = DEFAULT_LIMIT) -> ValidationResult:
    """Return the sanitized SELECT, or an error reason."""
    sql = _strip_fences(sql)
    if not sql:
        return ValidationResult(False, "", "Empty SQL")

    # Multiple statements -> reject.
    try:
        parsed_list = sqlglot.parse(sql, read="duckdb")
    except Exception as e:
        return ValidationResult(False, sql, f"Parse error: {e}")

    parsed_list = [p for p in parsed_list if p is not None]
    if len(parsed_list) != 1:
        return ValidationResult(False, sql, "Only a single statement is allowed")

    tree = parsed_list[0]

    # Walk the tree; reject any forbidden node.
    for node in tree.walk():
        n = node[0] if isinstance(node, tuple) else node
        if type(n) in FORBIDDEN:
            return ValidationResult(False, sql, f"Forbidden operation: {type(n).__name__}")

    # Must be a SELECT (or a CTE-wrapped SELECT).
    root = tree
    if not isinstance(root, (exp.Select, exp.With, exp.Subquery, exp.Union)):
        return ValidationResult(False, sql, f"Only SELECT queries allowed; got {type(root).__name__}")

    # Add LIMIT if missing (and not already an aggregate-only query without a LIMIT).
    final_select: exp.Select | None = None
    if isinstance(root, exp.Select):
        final_select = root
    elif isinstance(root, exp.With) and isinstance(root.this, exp.Select):
        final_select = root.this
    elif isinstance(root, exp.Union):
        # Add LIMIT to the Union itself
        if not root.args.get("limit"):
            root.set("limit", exp.Limit(expression=exp.Literal.number(default_limit)))

    if final_select is not None and not final_select.args.get("limit"):
        final_select.set(
            "limit", exp.Limit(expression=exp.Literal.number(default_limit))
        )

    sanitized = tree.sql(dialect="duckdb")
    return ValidationResult(True, sanitized, "")
