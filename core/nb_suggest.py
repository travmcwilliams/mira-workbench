"""
core/nb_suggest.py — quick-method pills + context-aware suggestions for the Notebook.

Two ways to get from a blank cell to a useful analysis:

1. **Method pills** (`METHOD_LIBRARY`) — a curated palette of common pandas /
   data-analysis snippets, grouped by task. Each is a template whose tokens
   (``{{NUM}}``, ``{{CAT}}``, ``{{TARGET}}``, ``{{DATETIME}}``) are filled from the
   live dataset, so clicking one inserts a ready-to-run cell. Fully deterministic,
   no model needed.

2. **AI suggestions** (`suggest`) — a lightweight RAG step: build a context doc
   from the dataset schema + target + the user's current notebook (code and the
   last output/error of each cell — "the screen"), retrieve the most relevant
   recipes from a local corpus (lexical overlap), and ask the chat model for a few
   tailored next steps grounded in those recipes. With no model it **degrades to
   the retrieved recipes**, so it always returns something runnable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# curated method palette (deterministic pills)
# --------------------------------------------------------------------------- #

# Tokens filled from the dataset: NUM=a numeric col, CAT=a categorical col,
# TARGET=the target (or a numeric col), DATETIME=a datetime col.
METHOD_LIBRARY: dict[str, list[dict[str, str]]] = {
    "Explore": [
        {"label": "Shape & dtypes", "code": "df.info()"},
        {"label": "Summary stats", "code": "df.describe(include='all').T"},
        {"label": "Missing %", "code": "df.isna().mean().sort_values(ascending=False)"},
        {"label": "Head", "code": "df.head(20)"},
        {"label": "Unique counts", "code": "df.nunique().sort_values(ascending=False)"},
    ],
    "Clean": [
        {"label": "Drop missing rows", "code": "df.dropna()"},
        {"label": "Fill numeric NaNs (median)",
         "code": "df.fillna(df.median(numeric_only=True))"},
        {"label": "Drop duplicates", "code": "df.drop_duplicates()"},
        {"label": "Coerce NUM to numeric",
         "code": "pd.to_numeric(df['{{NUM}}'], errors='coerce').describe()"},
    ],
    "Aggregate": [
        {"label": "Group by CAT (mean)",
         "code": "df.groupby('{{CAT}}').mean(numeric_only=True)"},
        {"label": "Value counts CAT", "code": "df['{{CAT}}'].value_counts()"},
        {"label": "Correlation matrix", "code": "df.corr(numeric_only=True)"},
        {"label": "Pivot CAT × mean NUM",
         "code": "df.pivot_table(index='{{CAT}}', values='{{NUM}}', aggfunc='mean')"},
    ],
    "Visualize": [
        {"label": "Histogram NUM",
         "code": "df['{{NUM}}'].plot.hist(bins=30); plt.title('{{NUM}}')"},
        {"label": "Scatter NUM vs TARGET",
         "code": "df.plot.scatter(x='{{NUM}}', y='{{TARGET}}', alpha=.4)"},
        {"label": "Correlation heatmap",
         "code": "px.imshow(df.corr(numeric_only=True), text_auto='.2f',\n"
                 "          aspect='auto', title='Correlation')"},
        {"label": "Box NUM by CAT",
         "code": "df.boxplot(column='{{NUM}}', by='{{CAT}}'); plt.suptitle('')"},
        {"label": "Bar: top CAT",
         "code": "df['{{CAT}}'].value_counts().head(15).plot.bar()"},
    ],
    "Stats / Model": [
        {"label": "Target distribution", "code": "df['{{TARGET}}'].describe()"},
        {"label": "Top correlations w/ TARGET",
         "code": "df.corr(numeric_only=True)['{{TARGET}}'].abs()"
                 ".sort_values(ascending=False).head(12)"},
        {"label": "Skew of numeric cols",
         "code": "df.select_dtypes('number').skew().sort_values()"},
        {"label": "Outliers (IQR) for NUM",
         "code": "q1, q3 = df['{{NUM}}'].quantile([.25, .75])\n"
                 "iqr = q3 - q1\n"
                 "mask = (df['{{NUM}}'] < q1 - 1.5*iqr) | (df['{{NUM}}'] > q3 + 1.5*iqr)\n"
                 "df[mask]"},
        {"label": "Quick baseline model",
         "code": "from sklearn.ensemble import RandomForestRegressor\n"
                 "from sklearn.model_selection import cross_val_score\n"
                 "num = df.select_dtypes('number').dropna()\n"
                 "y = num['{{TARGET}}']; X = num.drop(columns=['{{TARGET}}'])\n"
                 "cross_val_score(RandomForestRegressor(n_estimators=100),\n"
                 "                X, y, cv=3, scoring='r2')"},
    ],
}


# --------------------------------------------------------------------------- #
# recipe corpus for retrieval (superset of the palette + a few richer analyses)
# --------------------------------------------------------------------------- #


@dataclass
class Recipe:
    title: str
    keywords: str
    code: str


RECIPE_CORPUS: list[Recipe] = [
    Recipe("Profile the dataset", "overview shape dtypes info missing describe summary",
           "df.info()\ndf.describe(include='all').T"),
    Recipe("Missing-value audit", "missing null nan quality completeness",
           "df.isna().mean().sort_values(ascending=False)"),
    Recipe("Correlation with the target", "correlation relationship drivers target predictors",
           "df.corr(numeric_only=True)['{{TARGET}}'].abs()"
           ".sort_values(ascending=False).head(12)"),
    Recipe("Correlation heatmap", "correlation heatmap matrix multicollinearity",
           "px.imshow(df.corr(numeric_only=True), text_auto='.2f', aspect='auto')"),
    Recipe("Distribution of the target", "target distribution histogram balance skew",
           "df['{{TARGET}}'].plot.hist(bins=40); plt.title('{{TARGET}}')"),
    Recipe("Group-by aggregation", "group by category segment aggregate mean compare cohort",
           "df.groupby('{{CAT}}').agg({'{{NUM}}': ['mean', 'median', 'count']})"),
    Recipe("Category frequency", "value counts category frequency cardinality",
           "df['{{CAT}}'].value_counts().head(20)"),
    Recipe("Outlier detection (IQR)", "outliers anomalies iqr extreme robust",
           "q1, q3 = df['{{NUM}}'].quantile([.25, .75]); iqr = q3 - q1\n"
           "df[(df['{{NUM}}'] < q1-1.5*iqr) | (df['{{NUM}}'] > q3+1.5*iqr)]"),
    Recipe("Skew & transform check", "skew skewness log transform normal distribution",
           "df.select_dtypes('number').skew().sort_values()"),
    Recipe("Feature importance baseline", "model importance predict baseline random forest",
           "from sklearn.ensemble import RandomForestRegressor\n"
           "num = df.select_dtypes('number').dropna()\n"
           "y = num['{{TARGET}}']; X = num.drop(columns=['{{TARGET}}'])\n"
           "m = RandomForestRegressor(n_estimators=200).fit(X, y)\n"
           "import pandas as pd; pd.Series(m.feature_importances_, index=X.columns)"
           ".sort_values(ascending=False)"),
    Recipe("Time trend", "time series date trend over time temporal resample",
           "d = df.copy(); d['{{DATETIME}}'] = pd.to_datetime(d['{{DATETIME}}'], errors='coerce')\n"
           "d.set_index('{{DATETIME}}')['{{NUM}}'].resample('M').mean().plot()"),
    Recipe("Pairwise scatter of top numerics", "scatter pairplot relationship bivariate",
           "cols = df.select_dtypes('number').columns[:4]\n"
           "pd.plotting.scatter_matrix(df[cols], figsize=(8, 8))"),
]


# --------------------------------------------------------------------------- #
# column selection + templating
# --------------------------------------------------------------------------- #


def pick_columns(df: Any, schema: dict[str, str] | None = None,
                 target: str | None = None) -> dict[str, str]:
    """Choose representative NUM/CAT/DATETIME/TARGET columns for templating."""
    cols = [str(c) for c in getattr(df, "columns", [])]
    num, cat, dt = [], [], []
    if schema:
        for c, t in schema.items():
            if t == "numeric":
                num.append(c)
            elif t == "categorical":
                cat.append(c)
            elif t == "datetime":
                dt.append(c)
    if not num:
        try:
            num = list(df.select_dtypes("number").columns)
        except Exception:
            num = []
    if not dt:
        try:
            dt = list(df.select_dtypes("datetime").columns)
        except Exception:
            dt = []
    if not cat:
        cat = [c for c in cols if c not in num and c not in dt]

    def first(xs: list[str], fb: str) -> str:
        return xs[0] if xs else fb

    any_col = first(cols, "col")
    tgt = target if (target and target in cols) else first(num, any_col)
    return {
        "NUM": first(num, tgt),
        "CAT": first(cat, any_col),
        "DATETIME": first(dt, any_col),
        "TARGET": tgt,
    }


def fill(code: str, cols: dict[str, str]) -> str:
    """Replace {{TOKEN}} placeholders with concrete column names."""
    out = code
    for token, col in cols.items():
        out = out.replace("{{" + token + "}}", str(col))
    return out


def library_for(df: Any, schema: dict[str, str] | None = None,
                target: str | None = None) -> dict[str, list[dict[str, str]]]:
    """The method palette with every snippet filled for this dataset."""
    cols = pick_columns(df, schema, target)
    return {cat: [{"label": m["label"], "code": fill(m["code"], cols)} for m in items]
            for cat, items in METHOD_LIBRARY.items()}


# --------------------------------------------------------------------------- #
# retrieval (lexical RAG) + context
# --------------------------------------------------------------------------- #

_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")}


def retrieve(context: str, k: int = 5) -> list[Recipe]:
    """Top-k recipes by token overlap with the context (simple, offline RAG)."""
    ctx = _tokens(context)
    scored = []
    for i, r in enumerate(RECIPE_CORPUS):
        score = len(ctx & _tokens(r.keywords + " " + r.title))
        scored.append((score, i, r))
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [r for _, _, r in scored[:k]]


def build_context(schema: dict[str, str] | None, target: str | None,
                  task_type: str | None, n_rows: int, cells: list[Any],
                  results: dict[str, Any] | None = None) -> str:
    """A compact description of the dataset + the current notebook ('the screen')."""
    lines = [f"Dataset: {n_rows:,} rows.",
             f"Target: {target or '(none)'}  Task: {task_type or 'unknown'}"]
    if schema:
        lines.append("Columns: " + ", ".join(f"{c}({t})" for c, t in list(schema.items())[:40]))
    if cells:
        lines.append("\nCurrent notebook cells:")
        for i, cell in enumerate(cells[:12]):
            src = (getattr(cell, "source", "") or "").strip().replace("\n", " ")
            lines.append(f"  [{i + 1}] {src[:120]}")
            cid = str(getattr(cell, "id", "") or "")
            res = (results or {}).get(cid) if cid else None
            if res is not None:
                if getattr(res, "error", None):
                    lines.append(f"      ! error: {str(res.error).splitlines()[-1][:100]}")
                elif getattr(res, "stdout", ""):
                    lines.append(f"      out: {res.stdout.strip()[:100]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# suggestions (LLM grounded in retrieved recipes, with offline fallback)
# --------------------------------------------------------------------------- #

_SYS = (
    "You are a data-analysis copilot inside a notebook. Given the dataset context, "
    "the user's current notebook, and a set of candidate recipe templates, propose "
    "the most useful NEXT analysis steps. Reply with STRICT JSON only:\n"
    '[{"title": "short label", "code": "python for one cell"}]\n'
    "Rules: 3-6 items. Each `code` operates on `df` (pandas) and may use pd/np/plt/px. "
    "Prefer steps that build on what's already in the notebook and haven't been done "
    "yet. Ground them in the provided recipes when relevant; end a cell in a bare "
    "expression to display a result. No prose outside the JSON."
)


def _fallback(recipes: list[Recipe], cols: dict[str, str]) -> list[dict[str, str]]:
    return [{"title": r.title, "code": fill(r.code, cols)} for r in recipes]


def suggest(
    *,
    df: Any,
    schema: dict[str, str] | None,
    target: str | None,
    task_type: str | None,
    cells: list[Any],
    results: dict[str, Any] | None = None,
    chat_model: Any = None,       # core.chat.ChatModel or None
    runtime_key: str | None = None,
    n: int = 5,
) -> list[dict[str, str]]:
    """Return [{title, code}] next-step suggestions. Offline → retrieved recipes."""
    cols = pick_columns(df, schema, target)
    n_rows = int(getattr(df, "shape", [0])[0]) if getattr(df, "shape", None) else 0
    context = build_context(schema, target, task_type, n_rows, cells, results)
    recipes = retrieve(context, k=max(n, 5))

    if chat_model is None:
        return _fallback(recipes, cols)[:n]

    grounding = "\n".join(f"- {r.title}: {r.keywords}" for r in recipes)
    user = (f"{context}\n\nCandidate recipes:\n{grounding}\n\n"
            f"Available columns for templating: {cols}\n"
            f"Propose {n} next steps as JSON.")
    try:
        from core.chat import complete_chat
        raw = complete_chat(
            [{"role": "user", "content": user}],
            provider=chat_model.provider, model=chat_model.model,
            api_key=runtime_key, system=_SYS, max_tokens=1200,
        )
        items = _parse_suggestions(raw)
        if items:
            return items[:n]
    except Exception:
        pass
    return _fallback(recipes, cols)[:n]


def _parse_suggestions(raw: str) -> list[dict[str, str]]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    out: list[dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("code"):
            out.append({"title": str(item.get("title") or "Suggestion")[:80],
                        "code": str(item["code"])})
    return out
