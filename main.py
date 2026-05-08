"""
main.py — Talk-to-Data CLI
Run with: python main.py

Flow:
  1. User loads CSV/Parquet files (file, folder, or use existing)
  2. System scans all tables and builds a schema cache
  3. User asks natural language questions in a loop
  4. Question → full schema → Groq (70B) → SQL → DuckDB → chart in browser
"""
import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 output on Windows (emoji in print statements would otherwise crash
# with UnicodeEncodeError on cp1252 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db     import register_file, get_registered_tables, run_query, get_explain_plan, DATA_DIR
from schema import get_or_build, enrich_schema, get_schema_for_prompt, SCHEMA_CACHE_PATH, RETRIEVAL_THRESHOLD
from llm    import get_sql
from charts import show_chart, CHART_TYPES, pick_chart_type

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SUPPORTED_EXTS = {".csv", ".parquet"}

# ── Print helpers ─────────────────────────────────────────────────────────────
SEP  = "─" * 62
SEP2 = "═" * 62

def header(text):   print(f"\n{SEP2}\n  {text}\n{SEP2}")
def info(text):     print(f"  {text}")
def ok(text):       print(f"  ✅  {text}")
def warn(text):     print(f"  ⚠   {text}")
def err(text):      print(f"  ❌  {text}")
def ask(text):      return input(f"\n  {text} ").strip()


# ── Step 1: Load data files ───────────────────────────────────────────────────

def _ingest_path(raw: str) -> int:
    """
    Copy a file or every CSV/Parquet in a folder into data/.
    Returns number of files copied.
    """
    src = Path(raw.strip().strip('"').strip("'"))
    if not src.exists():
        err(f"Not found: {src}")
        return 0

    if src.is_dir():
        files = [f for f in sorted(src.iterdir())
                 if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
        if not files:
            warn(f"No CSV/Parquet files found in: {src}")
            return 0
        info(f"Folder — {len(files)} file(s) found:")
        for f in files:
            shutil.copy2(f, DATA_DIR / f.name)
            ok(f"  Copied → {f.name}")
        return len(files)

    if src.suffix.lower() not in SUPPORTED_EXTS:
        warn(f"Unsupported format '{src.suffix}' — use .csv or .parquet")
        return 0
    shutil.copy2(src, DATA_DIR / src.name)
    ok(f"Copied → {src.name}")
    return 1


def load_files() -> list[str]:
    header("STEP 1 — Load Data Files")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

    if existing:
        info(f"Files in data/ ({len(existing)}):")
        for p in existing:
            print(f"    • {p.name}")

    print()
    print("  Options:")
    print("    [1] Use files already in the data/ folder")
    print("    [2] Enter a file or folder path to load")
    print()
    choice = ask("Enter 1 or 2:").strip()

    if choice == "2":
        info("Enter a file or folder path. "
             "Press ENTER on a blank line when done.")
        print()
        total = 0
        while True:
            raw = ask("Path (or ENTER to finish):")
            if not raw:
                break
            total += _ingest_path(raw)
        if total == 0:
            warn("No files were copied.")

    # Register everything in data/
    files = sorted(
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        err("No data files found in data/. "
            "Add CSV or Parquet files and restart.")
        sys.exit(1)

    print()
    info("Registering with DuckDB:")
    tables = []
    for f in files:
        try:
            name = register_file(f)
            tables.append(name)
            ok(f"{f.name}  →  {name}")
        except Exception as e:
            err(f"Could not register {f.name}: {e}")

    if not tables:
        err("No files could be registered.")
        sys.exit(1)

    return tables


# ── Step 2: Build schema ──────────────────────────────────────────────────────

def build_schema(tables: list[str]) -> dict:
    header("STEP 2 — Building Schema")

    n = len(tables)
    mode = "full schema" if n <= RETRIEVAL_THRESHOLD else "TF-IDF retrieval"
    info(f"{n} table(s) detected → will use {mode} strategy")
    if n > RETRIEVAL_THRESHOLD:
        warn(f"Large schema ({n} tables). "
             f"Install scikit-learn for best retrieval: pip install scikit-learn")
    print()

    # ── Phase 1: structural scan (fast, always) ────────────────────────────
    if SCHEMA_CACHE_PATH.exists():
        import json
        with open(SCHEMA_CACHE_PATH) as f:
            cached_keys = set(json.load(f).keys())
        if cached_keys == set(tables):
            info("Schema cache found — loading...")
            choice = ask("Use cached schema? [Y/n]:").lower()
            if choice in ("", "y", "yes"):
                schema = get_or_build(tables, force_rebuild=False)
                enriched = sum(1 for e in schema.values() if e.get("enriched"))
                ok(f"Loaded: {len(schema)} tables ({enriched} with column descriptions)")
                return schema
        else:
            info("Cache is stale (tables changed) — rebuilding...")

    info("Phase 1: scanning table structure...")
    schema = get_or_build(tables, force_rebuild=True)
    print()
    ok(f"Phase 1 complete: {len(schema)} tables scanned")

    # ── Phase 2: LLM column descriptions (once, cached) ───────────────────
    if GROQ_API_KEY:
        print()
        info("Phase 2: generating column descriptions with Groq...")
        info("(Runs once per table, results cached — improves SQL accuracy)")
        print()
        schema = enrich_schema(schema, GROQ_API_KEY)
        from schema import save_schema
        save_schema(schema)
        enriched = sum(1 for e in schema.values() if e.get("enriched"))
        print()
        ok(f"Phase 2 complete: {enriched}/{len(schema)} tables described")
    else:
        warn("GROQ_API_KEY not set — skipping column descriptions.")
        warn("Set it to enable richer schema context for better SQL accuracy.")

    print()
    ok(f"Schema cached to {SCHEMA_CACHE_PATH}")
    return schema


# ── Schema display ────────────────────────────────────────────────────────────

def show_schema(schema: dict):
    header("LOADED TABLES")
    for name, entry in schema.items():
        cols = [c[0] for c in entry["columns"]]
        print(f"\n  📄 {name}  ({entry['row_count']:,} rows)")
        print(f"     {entry['description']}")
        print(f"     Columns: {', '.join(cols)}")


# ── Step 3: Question loop ─────────────────────────────────────────────────────

def question_loop(schema: dict):
    header("STEP 3 — Ask Questions")
    info("Ask anything about your data in plain English.")
    info("Commands:  schema | charts | history | quit")

    history = []

    while True:
        print(f"\n{SEP}")
        question = ask("Your question:").strip()
        if not question:
            continue

        q = question.lower().strip()

        # ── Built-in commands ──────────────────────────────────────────────
        if q in ("quit", "exit", "q"):
            print()
            info("Goodbye! Charts saved in charts/")
            break

        if q == "schema":
            show_schema(schema)
            continue

        if q == "charts":
            from charts import CHARTS_DIR
            saved = list(CHARTS_DIR.glob("*.html"))
            if saved:
                info(f"Saved charts ({len(saved)}):")
                for c in saved:
                    print(f"    • {c.name}")
            else:
                info("No charts saved yet.")
            continue

        if q == "history":
            if not history:
                info("No queries yet.")
            else:
                info(f"Query history ({len(history)}):")
                for i, h in enumerate(history, 1):
                    icon = "✅" if not h["error"] else "❌"
                    print(f"    {icon} {i}. {h['question']}")
            continue

        if not GROQ_API_KEY:
            err("GROQ_API_KEY not set.")
            info("Get a free key at https://console.groq.com")
            info("Then:  set GROQ_API_KEY=gsk_...  and restart.")
            continue

        # ── Pipeline ───────────────────────────────────────────────────────
        print()

        # 1. Get schema string (full or retrieved based on size)
        schema_str, mode = get_schema_for_prompt(question, schema)
        info(f"Schema mode: {mode}  "
             f"({len(schema)} tables total, "
             f"{schema_str.count('Table:') } sent to model)")

        # 2. Generate SQL
        info("Generating SQL...")
        try:
            sql, model = get_sql(question, schema_str, GROQ_API_KEY)
            info(f"Model: {model}")
            print(f"\n  Generated SQL:\n  {'─'*52}")
            for line in sql.splitlines():
                print(f"    {line}")
            print(f"  {'─'*52}")
        except RuntimeError as e:
            err(str(e))
            history.append({"question": question, "sql": "", "error": str(e)})
            continue

        # 3. Execute SQL
        df, sql_err = run_query(sql)
        if sql_err:
            err(f"SQL execution failed: {sql_err}")
            print()
            info("Available tables and columns:")
            for name, entry in schema.items():
                cols = [c[0] for c in entry["columns"]]
                print(f"    {name}: {', '.join(cols)}")
            info("Try rephrasing using the exact column names above.")
            history.append({"question": question, "sql": sql, "error": sql_err})
            continue

        if df is None or df.empty:
            warn("Query returned no rows.")
            history.append({"question": question, "sql": sql, "error": None})
            continue

        # 4. Get execution plan (EXPLAIN — logical plan, no re-execution)
        explain_plan = get_explain_plan(sql)

        # 5. Show result preview
        print()
        info(f"Result: {len(df):,} rows × {len(df.columns)} columns")
        print()
        print(df.head(8).to_string(index=False))
        if len(df) > 8:
            print(f"    ... ({len(df) - 8} more rows)")

        # 6. Chart type
        auto_type = pick_chart_type(df)
        print()
        info(f"Auto-detected chart: {auto_type}")
        chart_input = ask(
            f"Chart type [{'/'.join(CHART_TYPES)}] "
            f"(ENTER = keep '{auto_type}'):"
        ).lower()
        chart_type = chart_input if chart_input in CHART_TYPES else None

        # 7. Render and open (pass sql + explain_plan for the explain panel)
        try:
            path, _ = show_chart(df, question,
                                  sql=sql,
                                  explain_plan=explain_plan,
                                  chart_type=chart_type)
            print()
            ok(f"Chart saved → {path}")
            ok("Opened in your browser ✨")
            info("Tip: Click '🔍 Explain Query' in the chart to see the DuckDB execution plan")
        except Exception as e:
            err(f"Chart rendering failed: {e}")
            warn("Raw data shown above.")

        history.append({"question": question, "sql": sql, "error": None})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print()
    print(SEP2)
    print("  🗣  Talk-to-Data CLI")
    print("  Natural language → SQL → Chart")
    print(SEP2)

    if GROQ_API_KEY:
        ok(f"Groq API key: {GROQ_API_KEY[:12]}...")
    else:
        warn("GROQ_API_KEY not set.")
        warn("Get a free key at https://console.groq.com")
        warn("Then:  set GROQ_API_KEY=gsk_...  (no spaces, no quotes)")
        print()

    tables = load_files()
    schema = build_schema(tables)
    show_schema(schema)
    question_loop(schema)


if __name__ == "__main__":
    main()
