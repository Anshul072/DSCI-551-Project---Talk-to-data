"""
llm.py — Natural language → DuckDB SQL via Groq API.

Why Groq:
  • Free tier: 14,400 requests/day, no credit card needed
  • llama-3.3-70B: dramatically better instruction following than 7B models
  • Consistent, fast responses (Groq's custom inference chips)
  • OpenAI-compatible API — simple to use and swap out

Get a free key at: https://console.groq.com
"""
import re
from groq import Groq

# Model ladder — tried in order until one succeeds
# llama-3.3-70b is the primary: best reasoning, free tier
# llama-3.1-8b is a lightweight fallback if 70b hits rate limits
_MODEL_LADDER = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

_SYSTEM_MSG = """\
You are an expert DuckDB SQL analyst.

Rules:
1. Return ONLY a valid DuckDB SQL SELECT statement.
2. No explanation, no markdown fences, no backticks, no comments.
3. CRITICAL — use ONLY the exact column names listed in the schema.
   Do NOT invent column names. Do NOT assume columns exist.
4. ALWAYS wrap every column name in double quotes.
   This applies to ALL columns without exception, including normal names.
   Wrong:   SELECT Player, Year, 3P FROM ...
   Correct: SELECT "Player", "Year", "3P" FROM ...
   This prevents syntax errors on columns like 3P, TS%, WS/48, Date Of Accident.
5. For time filtering across tables, always JOIN to get date information.
   Example: to filter driver_standings by year, JOIN to races on raceId
   and use races."year" (if integer column exists) or YEAR(races."date").
6. DuckDB date functions (apply to actual DATE columns only):
   - YEAR(col)                — extract year integer
   - MONTH(col)               — extract month integer
   - DATE_TRUNC('month', col)  — truncate to month
   If a table already has an integer year column, use it directly.
7. Always alias aggregated columns (e.g. SUM("points") AS total_points).
8. Do NOT place a semicolon before LIMIT, ORDER BY, GROUP BY, or HAVING.
9. LIMIT 500 rows unless the user asks for more."""


def _clean_sql(raw: str) -> str:
    """Strip markdown fences, fix semicolons before clauses."""
    # Remove ```sql ... ``` or ``` ... ```
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)

    raw = raw.replace("`", "")

    # Fix: semicolon before LIMIT / ORDER BY / GROUP BY / HAVING
    raw = re.sub(
        r";\s*(LIMIT|ORDER\s+BY|GROUP\s+BY|HAVING)",
        lambda m: " " + m.group(1),
        raw, flags=re.IGNORECASE
    )

    return raw.strip().rstrip(";").strip()


def get_sql(question: str, schema: str, groq_api_key: str) -> tuple[str, str]:
    """
    Convert a natural language question to DuckDB SQL using Groq.
    Returns (sql, model_used).
    """
    user_msg = (
        f"DATABASE SCHEMA (use ONLY these exact column names):\n"
        f"{schema}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"IMPORTANT: Every column and table you use must exist in the schema above.\n"
        f"Return ONLY the SQL query. No explanation, no markdown."
    )

    client = Groq(api_key=groq_api_key)

    for model in _MODEL_LADDER:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            sql = _clean_sql(resp.choices[0].message.content)
            return sql, model
        except Exception as e:
            short = model.split("-")[0] + "-" + model.split("-")[1]
            print(f"  ⚠  {short} failed: {str(e)[:70]}")
            print(f"     → trying next model...")

    raise RuntimeError(
        "All Groq models failed.\n"
        "  • Check that GROQ_API_KEY is set correctly\n"
        "  • Get a free key at https://console.groq.com\n"
        "  • Check https://status.groq.com for outages"
    )
