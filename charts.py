"""
charts.py — Auto chart type detection, Plotly rendering, and DuckDB plan viewer.
"""
from pathlib import Path
import json, re
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

CHARTS_DIR = Path(__file__).parent / "charts"
CHART_TYPES = ["bar", "line", "pie", "scatter", "table"]


# ── Chart type detection ───────────────────────────────────────────────────────

def _is_date_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s): return True
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(s.dropna().head(5))
            return True
        except Exception: return False
    return False

def _is_year_col(s: pd.Series) -> bool:
    """Return True if 80%+ of non-null numeric values fall in the year range 1900-2100."""
    nums = pd.to_numeric(s, errors="coerce").dropna()
    if len(nums) == 0:
        return False
    in_range = ((nums >= 1900) & (nums <= 2100)).sum()
    return in_range / len(nums) >= 0.8

def _maybe_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect long-format multi-series data and pivot to wide format.

    Trigger conditions (all must hold):
      - Exactly 1 categorical column
      - Exactly 2 numeric columns
      - Categorical column has 2–15 distinct values
      - More rows than distinct category values (i.e., multiple x-values per series)

    Year detection: if one numeric column has 80%+ values in 1900–2100, it becomes
    the x-axis (pivot index); otherwise the first numeric column is used.

    Returns a reset-index wide DataFrame where all columns are numeric
    (year/x-axis first, then one column per series), or the original DataFrame
    unchanged if the trigger conditions are not met.
    """
    nums = df.select_dtypes(include="number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()

    if len(cats) != 1 or len(nums) != 2:
        return df

    cat_col   = cats[0]
    n_distinct = df[cat_col].nunique()

    if not (2 <= n_distinct <= 15):
        return df

    # Require multiple rows per category (long format, not a simple two-num table)
    if len(df) <= n_distinct:
        return df

    # Determine x-axis col: prefer year-like numeric, else first numeric
    x_col   = next((c for c in nums if _is_year_col(df[c])), nums[0])
    val_col = next(c for c in nums if c != x_col)

    try:
        pivoted = (
            df.pivot_table(index=x_col, columns=cat_col,
                           values=val_col, aggfunc="sum")
            .reset_index()
        )
        pivoted.columns.name = None   # drop MultiIndex residue
        return pivoted
    except Exception:
        return df

def pick_chart_type(df: pd.DataFrame) -> str:
    if df.empty: return "table"
    # Check for pivotable multi-series data first
    pivoted = _maybe_pivot(df)
    if pivoted is not df:
        return "line"
    nums = df.select_dtypes(include="number").columns.tolist()
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if cats and _is_date_like(df[cats[0]]) and nums:   return "line"
    # Two all-numeric columns where one is year-like → single-series time line
    if len(cats) == 0 and len(nums) == 2 and any(_is_year_col(df[c]) for c in nums):
        return "line"
    if len(cats)==1 and len(nums)==1 and len(df)<=8:   return "pie"
    if len(nums)>=2 and len(cats)<=1:                  return "scatter"
    if cats and nums:                                  return "bar"
    return "table"

def _prep(df):
    df = _maybe_pivot(df)   # Long-format → wide pivot BEFORE date coercion
    df = df.copy()
    for c in df.select_dtypes(exclude="number").columns:
        if _is_date_like(df[c]):
            try: df[c] = pd.to_datetime(df[c])
            except Exception: pass
    return df


# ── Chart builders ─────────────────────────────────────────────────────────────

def _bar(df, x, y):
    f = go.Figure()
    f.add_trace(go.Bar(x=df[x].tolist(), y=df[y].tolist(), name=y, marker_color="#1F4E79"))
    f.update_layout(xaxis_title=x, yaxis_title=y)
    return f

def _line(df, x, ys):
    f = go.Figure()
    colors = ["#1F4E79","#2E75B6","#4472C4","#70AD47","#ED7D31"]
    for i,y in enumerate(ys):
        f.add_trace(go.Scatter(x=df[x].tolist(), y=df[y].tolist(), mode="lines+markers",
            name=y, line=dict(color=colors[i%len(colors)], width=2), marker=dict(size=7)))
    f.update_layout(xaxis_title=x)
    return f

def _scatter(df, x, y):
    f = go.Figure()
    f.add_trace(go.Scatter(x=df[x].tolist(), y=df[y].tolist(), mode="markers", name=y,
        marker=dict(size=10, color="#1F4E79", opacity=0.8),
        text=df.apply(lambda r: "<br>".join(f"{c}: {r[c]}" for c in df.columns), axis=1).tolist(),
        hovertemplate="%{text}<extra></extra>"))
    f.update_layout(xaxis_title=x, yaxis_title=y)
    return f

def _pie(df, labels, values):
    f = go.Figure()
    f.add_trace(go.Pie(labels=df[labels].tolist(), values=df[values].tolist(), hole=0.35, textinfo="label+percent"))
    return f

def _table(df):
    return go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in df.columns],
                    fill_color="#1F4E79", font=dict(color="white",size=13), align="left", height=32),
        cells=dict(values=[df[c].astype(str).tolist() for c in df.columns],
                   fill_color=[["#F8F9FA","#FFFFFF"]*(len(df)//2+1)],
                   align="left", height=28, font=dict(size=12)))])

def _build_all(df):
    df = _prep(df)
    if df.empty: return {"table": _table(df)}
    nums  = df.select_dtypes(include="number").columns.tolist()
    cats  = df.select_dtypes(exclude="number").columns.tolist()
    x     = cats[0] if cats else (nums[0] if nums else df.columns[0])
    ynums = [c for c in nums if c != x] or nums
    figs  = {"table": _table(df)}
    if ynums:
        try: figs["bar"]     = _bar(df, x, ynums[0])
        except Exception: pass
        try: figs["line"]    = _line(df, x, ynums)
        except Exception: pass
        try:
            ys = ynums[1] if len(ynums)>1 else ynums[0]
            figs["scatter"]  = _scatter(df, x, ys)
        except Exception: pass
    if cats and nums and len(df)<=30:
        try: figs["pie"]     = _pie(df, cats[0], nums[0])
        except Exception: pass
    return figs


# ── Execution plan parser ──────────────────────────────────────────────────────

_GLOSSARY = {
    "TOP_N":               ("Top-N Sort",          "#60a5fa",
        "Retrieves the N highest/lowest rows without sorting the full dataset. "
        "More efficient than a full ORDER BY + LIMIT — DuckDB stops as soon as "
        "it has collected the top N rows."),
    "HASH_JOIN":           ("Hash Join",            "#c084fc",
        "Builds a hash table from the smaller (probe) table, then scans the larger "
        "table and probes the hash table for matches. DuckDB's default join strategy "
        "for INNER and LEFT joins on equality conditions."),
    "NESTED_LOOP_JOIN":    ("Nested Loop Join",     "#c084fc",
        "Compares every row from the outer table against every row of the inner table. "
        "Used for correlated subqueries. O(n²) — slower than Hash Join for large inputs."),
    "SEQ_SCAN":            ("Sequential Scan",      "#7dd3a8",
        "Reads data from the table column-by-column (columnar scan). Only the columns "
        "referenced in the query are read — this is DuckDB's projection pushdown, "
        "a core advantage of columnar storage over row-based databases."),
    "TABLE_SCAN":          ("Table Scan",           "#7dd3a8",
        "Reads data from a registered table or view using vectorized I/O. "
        "DuckDB processes rows in DataChunks of 1,024 rows to maximise CPU cache "
        "locality and enable SIMD acceleration."),
    "FILTER":              ("Filter — Predicate Pushdown", "#f87171",
        "Applies WHERE conditions as early in the plan as possible, eliminating rows "
        "before they reach expensive operators. This is predicate pushdown — "
        "a key query optimisation that avoids unnecessary computation."),
    "HASH_GROUP_BY":       ("Hash Group-By",        "#fb923c",
        "Groups rows using a hash table. DuckDB's vectorized aggregation engine "
        "processes 1,024 rows at a time (one DataChunk), maximising CPU cache "
        "hits and enabling SIMD instructions for SUM, COUNT, AVG etc."),
    "STREAMING_GROUP_BY":  ("Streaming Group-By",   "#fb923c",
        "Groups pre-sorted rows in a single pass without a hash table. "
        "More memory-efficient than Hash Group-By when the input is already sorted."),
    "PROJECTION":          ("Projection — Column Pruning", "#67e8f9",
        "Selects only the columns required downstream. DuckDB's columnar storage "
        "means skipped columns are never read from memory at all — this is "
        "column pruning / projection pushdown."),
    "ORDER_BY":            ("Order By",             "#60a5fa",
        "Sorts the result set. DuckDB uses a parallel radix sort that operates "
        "directly on compressed column vectors."),
    "LIMIT":               ("Limit",                "#a3e635",
        "Stops execution once N rows have been produced. Uses short-circuit "
        "evaluation — all upstream operators stop immediately when the limit is hit."),
    "AGGREGATE":           ("Aggregate",            "#fb923c",
        "Computes aggregate functions (SUM, COUNT, AVG, MIN, MAX) over groups. "
        "Executed in vectorized batches aligned to DuckDB's 1,024-row DataChunks."),
    "UNGROUPED_AGGREGATE": ("Full-Table Aggregate", "#fb923c",
        "Computes a single aggregate over the entire result set with no GROUP BY. "
        "DuckDB parallelises this across available CPU cores."),
    "CROSS_PRODUCT":       ("Cross Product",        "#f43f5e",
        "Produces every combination of rows (Cartesian product). Usually signals a "
        "missing JOIN condition in the WHERE clause."),
    "DISTINCT":            ("Distinct",             "#e879f9",
        "Eliminates duplicate rows using a hash set — equivalent to GROUP BY all "
        "columns with no aggregation."),
    "RESULT_COLLECTOR":    ("Result Collector",     "#94a3b8",
        "Assembles the final output rows for return to the client. This is always "
        "the root node of a DuckDB query plan."),
}

def _match_operator(op_name: str) -> tuple:
    """Return (display_name, color, explanation) for a DuckDB operator string."""
    key = op_name.upper().replace(" ", "_")
    for k, v in _GLOSSARY.items():
        if k in key:
            return v
    return (op_name, "#94a3b8",
            "DuckDB internal operator. See duckdb.org/docs/internals for details.")


def parse_explain_tree(raw_plan: str) -> list:
    """
    Parse DuckDB's EXPLAIN output (box-drawing ASCII tree) into a clean
    list of nodes:
      { depth, sibling, operator, display_name, color, details, explanation }

    The plan is a tree of boxes. Multiple boxes on the same horizontal row
    are sibling branches. Depth increases as we go down the tree.
    """
    if not raw_plan or not raw_plan.strip():
        return []

    lines = raw_plan.splitlines()

    # ── Split into horizontal bands ──────────────────────────────────────
    # A new band starts with a ┌ line. Bands are separated by rows that
    # contain └ (bottom of boxes) followed by ┌ (top of next level).
    bands = []
    current = []
    for line in lines:
        if '┌' in line and current:
            # If there's already a bottom row (└) in current, this ┌ starts
            # a new level. Otherwise it's a sibling box on the same level.
            has_bottom = any('└' in l for l in current)
            # Check whether this ┌ is at a new y-position
            # (i.e., the previous line was a bottom/connector)
            prev = current[-1] if current else ''
            if has_bottom and ('└' in prev or '┴' in prev or '┬' in prev):
                bands.append(current)
                current = []
        current.append(line)
    if current:
        bands.append(current)

    nodes = []

    for depth, band in enumerate(bands):
        if not band:
            continue

        # Find the top-row of this band (contains ┌)
        top_line = next((l for l in band if '┌' in l), '')
        if not top_line:
            continue

        # Each ┌ marks the left edge of a box
        box_lefts = [m.start() for m in re.finditer(r'┌', top_line)]
        box_rights = box_lefts[1:] + [len(top_line) + 60]

        for sib_idx, (left, right) in enumerate(zip(box_lefts, box_rights)):
            # Collect text content for this box column
            content = []
            for line in band:
                padded = line.ljust(right)
                col = padded[left:right]
                # Strip ALL box-drawing and whitespace chars
                text = re.sub(r'[┌┐└┘├┤┬┴┼─│╔╗╚╝╠╣╦╩╬═║]', ' ', col)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    content.append(text)

            if not content:
                continue

            # Remove pure-dash separator lines
            content = [l for l in content
                       if not re.fullmatch(r'[-─\s]+', l)]

            if not content:
                continue

            op_raw = content[0].strip()

            # Gather details: clean up, merge split table names
            raw_details = content[1:]
            details = []
            skip_next = False
            for i, line in enumerate(raw_details):
                if skip_next:
                    skip_next = False
                    continue
                line = line.strip()
                if not line or re.fullmatch(r'[-─\s]+', line):
                    continue
                # Merge "memory.main" + ".tablename" into one line
                if (line == 'memory.main' and i+1 < len(raw_details)
                        and raw_details[i+1].strip().startswith('.')):
                    merged = 'memory.main' + raw_details[i+1].strip()
                    # Make table name readable
                    merged = merged.replace('memory.main.', '')
                    details.append(f"Table: {merged}")
                    skip_next = True
                    continue
                # Clean up "Table: memory.main.xxx" → "Table: xxx"
                line = re.sub(r'memory\.main\.', '', line)
                line = re.sub(r'memory\.main', '', line).strip()
                if line:
                    details.append(line)

            display_name, color, explanation = _match_operator(op_raw)

            nodes.append({
                "depth":        depth,
                "sibling":      sib_idx,
                "operator":     op_raw,
                "display_name": display_name,
                "color":        color,
                "details":      details,
                "explanation":  explanation,
            })

    return nodes


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"><title>{TITLE}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:Arial,sans-serif;background:#f5f7fa}

    .header{background:#1F4E79;color:white;padding:14px 24px}
    .header h2{font-size:16px;font-weight:600}
    .header p{font-size:12px;opacity:.75;margin-top:4px}

    .toolbar{display:flex;align-items:center;gap:8px;padding:12px 24px;
             background:white;border-bottom:1px solid #e0e0e0;flex-wrap:wrap}
    .toolbar span{font-size:13px;color:#555;margin-right:4px}
    .divider{width:1px;height:24px;background:#ddd;margin:0 6px}

    .btn{padding:6px 16px;border:1.5px solid #1F4E79;border-radius:20px;
         background:white;color:#1F4E79;font-size:13px;cursor:pointer;transition:all .15s}
    .btn:hover{background:#e8f0fe}
    .btn.active{background:#1F4E79;color:white}
    .btn-explain{border-color:#7B2D8B;color:#7B2D8B}
    .btn-explain:hover{background:#f3e5f5}
    .btn-explain.active{background:#7B2D8B;color:white}

    #main{display:flex;height:calc(100vh - 118px)}
    #chart-area{flex:1;min-width:0}

    /* ── Explain panel ── */
    #xpanel{width:0;overflow:hidden;transition:width .3s ease;
            background:#0f1117;color:#e2e8f0;font-family:"Courier New",monospace;
            display:flex;flex-direction:column;border-left:1px solid #222}
    #xpanel.open{width:520px;min-width:340px}

    .xp-head{padding:14px 16px;background:#0a0d14;border-bottom:1px solid #1e2a3a;
             display:flex;justify-content:space-between;align-items:center}
    .xp-head h3{font-size:14px;color:#a78bfa;letter-spacing:.5px;font-family:Arial,sans-serif}
    .xp-head small{font-size:11px;color:#4a5568;font-family:Arial,sans-serif}

    .xp-body{flex:1;overflow-y:auto;padding:14px 12px}
    .xp-body::-webkit-scrollbar{width:5px}
    .xp-body::-webkit-scrollbar-thumb{background:#2d3748;border-radius:3px}

    /* SQL block */
    .sql-box{background:#111827;border:1px solid #1e3a2a;border-radius:6px;
             padding:10px 12px;margin-bottom:14px;font-size:11px;
             line-height:1.8;color:#6ee7b7;white-space:pre-wrap;word-break:break-all}
    .sql-label{font-size:10px;color:#4a5568;font-family:Arial,sans-serif;
               margin-bottom:6px;letter-spacing:.8px}

    /* Tree */
    .tree{position:relative}
    .tree-node{margin:4px 0;position:relative}

    /* Node card */
    .node-card{border-radius:6px;border:1px solid transparent;
               cursor:pointer;transition:all .15s;overflow:hidden}
    .node-card:hover{filter:brightness(1.1)}

    .node-head{display:flex;align-items:center;gap:8px;padding:8px 12px}
    .node-icon{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .node-op{font-size:12px;font-weight:700;font-family:Arial,sans-serif;flex:1}
    .node-badge{font-size:10px;opacity:.7;white-space:nowrap;font-family:Arial,sans-serif}

    /* Details chips */
    .node-details{padding:0 12px 8px;display:flex;flex-wrap:wrap;gap:4px}
    .chip{background:rgba(0,0,0,.3);border-radius:4px;padding:2px 7px;
          font-size:10px;color:#94a3b8;font-family:Arial,sans-serif;
          border:1px solid rgba(255,255,255,.08)}

    /* Explanation */
    .node-explain{padding:8px 12px;border-top:1px solid rgba(255,255,255,.06);
                  font-size:11.5px;line-height:1.7;color:#94a3b8;
                  font-family:Arial,sans-serif;display:none}
    .node-card.open .node-explain{display:block}

    /* Tree connector lines */
    .connector{display:flex;margin-left:16px}
    .connector-line{width:2px;background:#1e2a3a;flex-shrink:0;margin-right:10px}

    /* Depth indentation */
    .depth-0{margin-left:0}
    .depth-1{margin-left:24px}
    .depth-2{margin-left:48px}
    .depth-3{margin-left:72px}
    .depth-4{margin-left:96px}
    .depth-sib{margin-left:24px;margin-top:6px}
  </style>
</head>
<body>

<div class="header">
  <h2>{TITLE}</h2>
  <p>{ROWS} rows &nbsp;·&nbsp; {COLS} columns &nbsp;·&nbsp; auto-detected: <b>{AUTO}</b></p>
</div>

<div class="toolbar">
  <span>Chart type:</span>
  {BUTTONS}
  <div class="divider"></div>
  <button class="btn btn-explain" id="btnExplain" onclick="toggleExplain()">
    🔍 Explain Query
  </button>
</div>

<div id="main">
  <div id="chart-area"></div>
  <div id="xpanel">
    <div class="xp-head">
      <h3>⚡ DuckDB Execution Plan</h3>
      <small>Click any node to see what it does</small>
    </div>
    <div class="xp-body" id="xbody"></div>
  </div>
</div>

<script>
const CHARTS = {CHARTS_JSON};
const NODES  = {NODES_JSON};
const SQL    = {SQL_JSON};

const BASE = {
  margin:{t:40,b:60,l:70,r:40},
  paper_bgcolor:"#fff",plot_bgcolor:"#f9f9f9",
  font:{family:"Arial",size:13},
  xaxis:{showgrid:true,gridcolor:"#eee",zeroline:false},
  yaxis:{showgrid:true,gridcolor:"#eee",zeroline:false},
};

let cur = "{DEFAULT}", xopen = false;

function show(t) {
  if (!CHARTS[t]) return;
  cur = t;
  document.querySelectorAll(".btn:not(.btn-explain)").forEach(b =>
    b.classList.toggle("active", b.dataset.type===t));
  const fig = CHARTS[t];
  Plotly.react("chart-area", fig.data,
    Object.assign({}, BASE, fig.layout||{}), {responsive:true});
}

function toggleExplain() {
  xopen = !xopen;
  document.getElementById("xpanel").classList.toggle("open", xopen);
  document.getElementById("btnExplain").classList.toggle("active", xopen);
  if (xopen && !document.getElementById("xbody").hasChildNodes()) buildTree();
  setTimeout(() => Plotly.Plots.resize("chart-area"), 320);
}

function buildTree() {
  const body = document.getElementById("xbody");

  // SQL block
  const sqlDiv = document.createElement("div");
  sqlDiv.innerHTML = `<div class="sql-label">GENERATED SQL</div>
    <div class="sql-box">${escHtml(SQL)}</div>`;
  body.appendChild(sqlDiv);

  if (!NODES || NODES.length === 0) {
    body.innerHTML += '<p style="color:#4a5568;font-size:12px;padding:8px">No plan available.</p>';
    return;
  }

  const treeDiv = document.createElement("div");
  treeDiv.className = "tree";

  NODES.forEach((node, i) => {
    const depthClass = node.sibling > 0 ? "depth-sib" : `depth-${Math.min(node.depth,4)}`;
    const card = document.createElement("div");
    card.className = `tree-node ${depthClass}`;

    // Connector line for non-root nodes
    const prefix = node.depth > 0 ? (node.sibling > 0 ? "├─ " : "└─ ") : "";

    // Detail chips — only meaningful lines
    const chips = (node.details || [])
      .filter(d => d && d.length > 0 && d.length < 60)
      .slice(0, 5)
      .map(d => `<span class="chip">${escHtml(d)}</span>`)
      .join("");

    card.innerHTML = `
      <div class="node-card" style="background:${colorBg(node.color)};border-color:${node.color}22"
           onclick="this.classList.toggle('open')">
        <div class="node-head">
          <div class="node-icon" style="background:${node.color}"></div>
          <span class="node-op" style="color:${node.color}">
            ${prefix}${escHtml(node.display_name)}
          </span>
          <span class="node-badge">${escHtml(node.operator)}</span>
        </div>
        ${chips ? `<div class="node-details">${chips}</div>` : ''}
        <div class="node-explain">${escHtml(node.explanation)}</div>
      </div>`;

    treeDiv.appendChild(card);
  });

  body.appendChild(treeDiv);
}

function colorBg(hex) {
  // Convert hex to very dark transparent background
  return hex + "18";
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

show(cur);
</script>
</body>
</html>"""


# ── Public entry point ─────────────────────────────────────────────────────────

def show_chart(df: pd.DataFrame, question: str,
               sql: str = "", explain_plan: str = "",
               chart_type: str = None) -> tuple[str, str]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    auto_type    = pick_chart_type(df)
    default_type = chart_type if chart_type in CHART_TYPES else auto_type
    figures      = _build_all(df)
    if default_type not in figures:
        default_type = auto_type if auto_type in figures else list(figures.keys())[0]

    def ser(fig):
        fig.update_layout(margin=dict(t=40,b=60,l=70,r=40),
                          paper_bgcolor="#fff", plot_bgcolor="#f9f9f9")
        return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))

    charts_j = {ct: ser(fig) for ct, fig in figures.items()}
    nodes    = parse_explain_tree(explain_plan) if explain_plan else []

    labels = {"bar":"📊 Bar","line":"📈 Line","pie":"🥧 Pie",
               "scatter":"⚬ Scatter","table":"📋 Table"}
    btns = "\n  ".join(
        f'<button class="btn{" active" if ct==default_type else ""}" '
        f'data-type="{ct}" onclick="show(\'{ct}\')">{labels[ct]}</button>'
        for ct in ["bar","line","scatter","pie","table"] if ct in figures)

    safe = question[:80].replace("'", "\\'")
    html = (_HTML
        .replace("{TITLE}",       safe)
        .replace("{ROWS}",        f"{len(df):,}")
        .replace("{COLS}",        str(len(df.columns)))
        .replace("{AUTO}",        auto_type)
        .replace("{BUTTONS}",     btns)
        .replace("{CHARTS_JSON}", json.dumps(charts_j))
        .replace("{NODES_JSON}",  json.dumps(nodes))
        .replace("{SQL_JSON}",    json.dumps(sql))
        .replace('"{DEFAULT}"',   f'"{default_type}"'))

    slug = "".join(c if c.isalnum() else "_" for c in question[:50].lower()).strip("_")
    out  = CHARTS_DIR / f"{slug}.html"
    out.write_text(html, encoding="utf-8")

    import webbrowser
    webbrowser.open(out.resolve().as_uri())
    return str(out), auto_type
