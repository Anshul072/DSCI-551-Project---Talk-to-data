"""
db.py — DuckDB connection, file registration, query execution.

Registration uses a 3-attempt strategy to handle malformed CSV files:
  Attempt 1: read_csv_auto  (fast, works for most files)
  Attempt 2: pandas         (handles encoding issues, BOM, bad quoting)
  Attempt 3: read_csv with explicit sep/quote settings

After registration, VARCHAR columns that contain date-like values are
automatically cast to DATE so YEAR(), MONTH(), DATE_TRUNC() work correctly.
"""
from pathlib import Path
import duckdb
import pandas as pd

_con: duckdb.DuckDBPyConnection | None = None
DATA_DIR = Path(__file__).parent / "data"

# Date patterns DuckDB can parse natively
_DATE_PATTERNS = [
    "%Y-%m-%d",       # 2018-03-25
    "%d/%m/%Y",       # 25/03/2018
    "%m/%d/%Y",       # 03/25/2018
    "%Y/%m/%d",       # 2018/03/25
    "%d-%m-%Y",       # 25-03-2018
]


def get_connection() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(":memory:")
    return _con


def _col_count(table_name: str, con) -> int:
    r = con.execute(
        f"SELECT COUNT(*) FROM information_schema.columns "
        f"WHERE table_name = '{table_name}'"
    ).fetchone()
    return r[0] if r else 0


def _is_collapsed_header(table_name: str, con) -> bool:
    cols = con.execute(
        f"SELECT column_name FROM information_schema.columns "
        f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
    ).fetchall()
    return len(cols) == 1 and "," in cols[0][0]


def _fix_date_columns(table_name: str, con):
    """
    Inspect every VARCHAR column. If its values parse as dates,
    ALTER the column to DATE type.
    Prevents YEAR(col) errors like:
      'No function matches year(VARCHAR)'
    """
    cols = con.execute(
        f"SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
    ).fetchall()

    fixed = []
    for col_name, dtype in cols:
        if dtype.upper() not in ("VARCHAR", "TEXT", "STRING"):
            continue

        # Sample up to 10 non-null values (quick existence check)
        try:
            samples = con.execute(
                f"SELECT \"{col_name}\" FROM {table_name} "
                f"WHERE \"{col_name}\" IS NOT NULL "
                f"AND \"{col_name}\" != '' LIMIT 10"
            ).fetchall()
        except Exception:
            continue

        if not samples:
            continue

        # Try each date pattern — if ≥ 95% of values parse, cast the column
        for pattern in _DATE_PATTERNS:
            try:
                # TRY_STRPTIME returns NULL on failure (strptime would throw)
                total = con.execute(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE \"{col_name}\" IS NOT NULL AND \"{col_name}\" != ''"
                ).fetchone()[0]
                if total == 0:
                    continue
                matched = con.execute(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE \"{col_name}\" IS NOT NULL AND \"{col_name}\" != '' "
                    f"AND TRY_STRPTIME(\"{col_name}\", '{pattern}') IS NOT NULL"
                ).fetchone()[0]
                # Cast if ≥ 95% of non-null values parse successfully
                if matched > 0 and matched / total >= 0.95:
                    con.execute(
                        f"ALTER TABLE {table_name} "
                        f"ALTER COLUMN \"{col_name}\" "
                        f"TYPE DATE USING TRY_CAST(\"{col_name}\" AS DATE)"
                    )
                    fixed.append(col_name)
                    break
            except Exception:
                continue

    if fixed:
        print(f"    ℹ  Auto-cast to DATE: {', '.join(fixed)}")


def register_file(filepath: str | Path) -> str:
    """
    Register a CSV or Parquet file as a DuckDB TABLE.
    Handles BOM/encoding issues and auto-casts date columns.
    Returns the table name.
    """
    path = Path(filepath)
    table_name = (
        path.stem.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    con = get_connection()

    # ── Parquet ───────────────────────────────────────────────────────────
    if path.suffix.lower() == ".parquet":
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_parquet('{path}')"
        )
        n = _col_count(table_name, con)
        print(f"    ✅ {path.name} → {table_name} ({n} columns, parquet)")
        return table_name

    # ── CSV: 3-attempt strategy ───────────────────────────────────────────

    registered = False

    # Attempt 1: DuckDB auto-detect
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{path}')"
        )
        if not _is_collapsed_header(table_name, con):
            registered = True
        else:
            print(f"    ⚠  {path.name}: collapsed header, retrying...")
    except Exception as e:
        print(f"    ⚠  {path.name}: auto-detect failed "
              f"({str(e)[:50]}), retrying...")

    # Attempt 2: pandas with BOM-safe encoding
    if not registered:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            if len(df.columns) > 1:
                con.execute(
                    f"CREATE OR REPLACE TABLE {table_name} "
                    f"AS SELECT * FROM df"
                )
                registered = True
            else:
                print(f"    ⚠  pandas also got 1 column, retrying...")
        except Exception as e:
            print(f"    ⚠  pandas failed ({str(e)[:50]}), retrying...")

    # Attempt 3: explicit separators
    if not registered:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(
                    path, sep=sep,
                    encoding="utf-8-sig", low_memory=False
                )
                if len(df.columns) > 1:
                    con.execute(
                        f"CREATE OR REPLACE TABLE {table_name} "
                        f"AS SELECT * FROM df"
                    )
                    registered = True
                    print(f"    ℹ  Used separator '{sep}'")
                    break
            except Exception:
                continue

    if not registered:
        print(f"    ❌ {path.name}: could not parse. "
              f"Check encoding and delimiter.")
        try:
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                f"SELECT * FROM read_csv_auto('{path}')"
            )
        except Exception:
            pass

    # ── Auto-cast VARCHAR date columns → DATE ─────────────────────────────
    _fix_date_columns(table_name, con)

    n = _col_count(table_name, con)
    print(f"    ✅ {path.name} → {table_name} ({n} columns)")
    return table_name


def get_registered_tables() -> list[str]:
    con = get_connection()
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_type IN ('VIEW', 'BASE TABLE') "
        "ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def run_query(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return get_connection().execute(sql).df(), None
    except Exception as e:
        return None, str(e)

def get_explain_plan(sql: str) -> str:
    """
    Run EXPLAIN on a SQL query and return the logical plan text.
    Uses EXPLAIN (not EXPLAIN ANALYZE) so the query is not re-executed.
    Returns the plan string, or an error message if it fails.
    """
    try:
        rows = get_connection().execute(f"EXPLAIN {sql}").fetchall()
        return "\n".join(row[1] for row in rows)
    except Exception as e:
        return f"Could not generate execution plan: {e}"
