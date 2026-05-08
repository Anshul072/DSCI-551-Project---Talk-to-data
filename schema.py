"""
schema.py — Builds and caches table metadata for all registered tables.

Two-phase build:
  Phase 1 (fast, always):  scan column names + sample values → keywords
  Phase 2 (once, optional): Groq LLM call per table → table description
                             + plain-English description for every column

The column descriptions are embedded in the schema string sent to the LLM,
giving it semantic context beyond just column names and types.

Example without descriptions:
  raceId   (BIGINT)
  year     (BIGINT)
  round    (BIGINT)

Example with descriptions:
  raceId   (BIGINT)   — unique identifier for each race
  year     (BIGINT)   — season year (e.g. 2023)
  round    (BIGINT)   — race number within the season (1 = first race)
"""
import re
import json
import time
from pathlib import Path

from db import get_connection

SCHEMA_CACHE_PATH = Path(__file__).parent / "registry" / "schema_cache.json"
RETRIEVAL_THRESHOLD = 50


# ── Phase 1: structural metadata ──────────────────────────────────────────────

def _col_keywords(columns: list) -> set:
    keywords = set()
    for col in columns:
        col = re.sub(r"([A-Z])", r"_\1", col).lower()
        keywords.update(w for w in re.split(r"[_\s\-]+", col) if len(w) > 2)
    return keywords


def _value_keywords(table_name: str) -> set:
    con = get_connection()
    keywords = set()
    cols = con.execute(
        f"SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
    ).fetchall()
    for col_name, dtype in cols:
        if dtype.upper() in ("VARCHAR", "TEXT", "STRING", "CHAR"):
            try:
                vals = con.execute(
                    f"SELECT DISTINCT {col_name} FROM {table_name} "
                    f"WHERE {col_name} IS NOT NULL LIMIT 20"
                ).fetchall()
                for (v,) in vals:
                    v = str(v).strip()
                    if not v or v.isdigit() or len(v) > 40:
                        continue
                    keywords.update(
                        w for w in re.split(r"[\s_\-/&]+", v.lower())
                        if len(w) > 2 and not w.isdigit()
                    )
            except Exception:
                pass
    return keywords


def build_entry(table_name: str) -> dict:
    """Phase 1: fast structural scan — no LLM."""
    con = get_connection()
    cols = con.execute(
        f"SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
    ).fetchall()
    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    col_names = [c for c, _ in cols]
    keywords  = _col_keywords(col_names) | _value_keywords(table_name)

    return {
        "description": f"Table containing {table_name.replace('_', ' ')} data",
        "keywords":    sorted(keywords),
        # columns stored as [name, type, description]
        # description is "" until Phase 2 enrichment runs
        "columns":     [[c, t, ""] for c, t in cols],
        "row_count":   row_count,
        "enriched":    False,
    }


def build_schema(tables: list) -> dict:
    """Phase 1 for all tables — fast, no LLM."""
    schema = {}
    for name in tables:
        print(f"  Scanning {name}...", end=" ", flush=True)
        t0 = time.time()
        schema[name] = build_entry(name)
        print(f"{len(schema[name]['columns'])} cols  [{time.time()-t0:.1f}s]")
    return schema


# ── Phase 2: LLM enrichment ───────────────────────────────────────────────────

_ENRICH_SYSTEM = """\
You are a data dictionary assistant. Given a database table schema and sample
rows, return ONLY valid JSON — no prose, no markdown fences."""

_ENRICH_USER = """\
Table: {table_name}
Columns: {columns}
Sample rows:
{sample_rows}

Return JSON in exactly this format:
{{
  "description": "One sentence describing what this table contains.",
  "columns": {{
    "col_name": "short plain-English description of what this column means",
    ...one entry per column...
  }}
}}

Rules:
- Keep each column description under 12 words
- Use plain English, not technical jargon
- For ID columns: explain what entity they identify
- For flag/status columns: list the possible values if obvious from samples
- For date columns: mention the format if relevant"""


def _extract_json(raw: str) -> dict | None:
    """
    Robustly extract JSON from model output.
    Handles markdown fences, leading prose, and trailing text.
    """
    # 1. Strip markdown fences first
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    # 2. Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 3. Find the outermost { ... } block and try parsing that
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _enrich_table(table_name: str, groq_api_key: str) -> dict:
    """
    Call Groq once for this table and return:
      { "description": str, "columns": { col_name: description, ... } }
    """
    from groq import Groq

    con = get_connection()
    cols = con.execute(
        f"SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
    ).fetchall()
    col_str    = ", ".join(f"{c} ({t})" for c, t in cols)
    sample_str = con.execute(
        f"SELECT * FROM {table_name} LIMIT 4"
    ).df().to_string(index=False)

    user_msg = _ENRICH_USER.format(
        table_name  = table_name,
        columns     = col_str,
        sample_rows = sample_str,
    )

    client = Groq(api_key=groq_api_key)

    # Try 70B first (more reliable JSON output), fall back to 8B
    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            resp = client.chat.completions.create(
                model    = model,
                messages = [
                    {"role": "system", "content": _ENRICH_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 800,
            )
            raw    = resp.choices[0].message.content.strip()
            result = _extract_json(raw)

            if result and "description" in result and "columns" in result:
                return result

            print(f"    ⚠  {model}: could not parse JSON, trying next...")

        except Exception as e:
            print(f"    ⚠  {model}: {str(e)[:60]}, trying next...")

    raise RuntimeError(f"All models failed enriching {table_name}")


def enrich_schema(schema: dict, groq_api_key: str) -> dict:
    """
    Phase 2: enrich all tables with LLM-generated descriptions.
    Returns an updated schema dict (modifies in place and returns it).

    Skips tables that are already enriched (so incremental enrichment works).
    """
    tables_to_enrich = [
        name for name, entry in schema.items()
        if not entry.get("enriched")
    ]

    if not tables_to_enrich:
        print("  All tables already enriched — skipping.")
        return schema

    print(f"  Enriching {len(tables_to_enrich)} table(s) with Groq "
          f"(llama-3.1-8b-instant)...")
    print("  This runs once and is cached. Future runs load from disk.\n")

    for name in tables_to_enrich:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = _enrich_table(name, groq_api_key)

            # Update table description
            schema[name]["description"] = result["description"]

            # Update column descriptions
            col_descs = result.get("columns", {})
            updated_cols = []
            for col_entry in schema[name]["columns"]:
                # col_entry is [name, type, description]
                col_name = col_entry[0]
                col_type = col_entry[1]
                desc     = col_descs.get(col_name, "")
                updated_cols.append([col_name, col_type, desc])
            schema[name]["columns"]  = updated_cols
            schema[name]["enriched"] = True

            n_described = sum(1 for c in updated_cols if c[2])
            print(f"✅  {n_described}/{len(updated_cols)} cols described  "
                  f"[{time.time()-t0:.1f}s]")

        except Exception as e:
            print(f"⚠   failed: {e}")

    return schema


# ── Persistence ────────────────────────────────────────────────────────────────

def save_schema(schema: dict, path: Path = SCHEMA_CACHE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema(path: Path = SCHEMA_CACHE_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # columns stored as [name, type, desc] — keep as lists (no tuple conversion)
    return data


def get_or_build(tables: list, force_rebuild: bool = False) -> dict:
    """Phase 1 only — fast structural scan."""
    if not force_rebuild:
        cached = load_schema()
        if cached and set(cached.keys()) == set(tables):
            return cached
    schema = build_schema(tables)
    save_schema(schema)
    return schema


# ── Schema string builders ─────────────────────────────────────────────────────

def _quote_col(name: str) -> str:
    """Quote column names that are not plain SQL identifiers.
    Covers: names starting with a digit (3P, 2PA), names with spaces
    (Date Of Accident), and names with special chars (TS%, WS/48, FG%).
    """
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        return name
    return f'"{name}"'


def build_full_schema_string(schema: dict) -> str:
    """
    Build the schema string sent to the SQL-generation model.

    Format WITH column descriptions (after enrichment):
      Table: races  (1,149 rows)
      About: Stores information about each Formula One race.
        raceId   (BIGINT)   — unique identifier for each race
        year     (BIGINT)   — season year (e.g. 2023)
        round    (BIGINT)   — race number within the season

    Format WITHOUT descriptions (before enrichment):
      Table: races  (1,149 rows)
      About: Table containing races data
        raceId   (BIGINT)
        year     (BIGINT)
    """
    parts = []
    for name, entry in schema.items():
        col_lines = []
        for col_entry in entry["columns"]:
            # Support both old [name, type] and new [name, type, desc] formats
            col_name = col_entry[0]
            col_type = col_entry[1]
            col_desc = col_entry[2] if len(col_entry) > 2 else ""

            quoted = _quote_col(col_name)
            if col_desc:
                col_lines.append(f"{quoted}  ({col_type})   — {col_desc}")
            else:
                col_lines.append(f"{quoted}  ({col_type})")

        parts.append(
            f"Table: {name}  ({entry['row_count']:,} rows)\n"
            f"About: {entry['description']}\n"
            + "\n".join(f"  {l}" for l in col_lines)
        )
    return "\n\n".join(parts)


def build_retrieved_schema_string(question: str, schema: dict,
                                   top_n: int = 5) -> str:
    """TF-IDF retrieval for large schemas (50+ tables)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        return _keyword_retrieval(question, schema, top_n)

    docs = {}
    for name, entry in schema.items():
        col_text = " ".join(c[0] for c in entry["columns"])
        kw_text  = " ".join(entry.get("keywords", [])[:30])
        docs[name] = f"{entry['description']} {col_text} {kw_text}"

    names  = list(docs.keys())
    corpus = list(docs.values())
    vec    = TfidfVectorizer(stop_words="english")
    mat    = vec.fit_transform(corpus)
    qvec   = vec.transform([question])
    scores = cosine_similarity(qvec, mat).flatten()

    top_idx  = np.argsort(scores)[::-1][:top_n]
    selected = [names[i] for i in top_idx if scores[i] > 0] or names[:top_n]
    return build_full_schema_string({n: schema[n] for n in selected})


def _keyword_retrieval(question: str, schema: dict, top_n: int) -> str:
    words = {
        w for w in re.split(r"[\s,\.?!;:()\-]+", question.lower())
        if len(w) > 2 and not w.isdigit()
    }
    scores = {
        name: len(words & set(entry.get("keywords", [])))
        for name, entry in schema.items()
    }
    ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [n for n, _ in ranked[:top_n]] or list(schema.keys())[:top_n]
    return build_full_schema_string({n: schema[n] for n in selected})


def get_schema_for_prompt(question: str, schema: dict) -> tuple:
    if len(schema) <= RETRIEVAL_THRESHOLD:
        return build_full_schema_string(schema), "full"
    else:
        return build_retrieved_schema_string(question, schema), "retrieved"
