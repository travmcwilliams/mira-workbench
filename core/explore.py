"""
core/explore.py

Notebook-style dataset exploration. Each "recipe" is a self-contained code
snippet that produces the chart most useful for one analysis concept; the engine
executes it to render a figure for the Streamlit notebook UI and emits the same
code as cells in an exportable .ipynb.

The recipe set mirrors the linear-algebra / distributions / unsupervised-learning
concepts:
  covariance matrix · eigenvectors & eigenvalues · PCA dimensionality reduction ·
  correlation · distance & scaling · k-means · Gaussian mixtures · hierarchical
  clustering (dendrogram) · DBSCAN · distribution fit · power-law · KL divergence ·
  feature graph (matrix representation of a graph).

Code strings are self-contained (they import their own libs and assume a
DataFrame `df`) so the exported notebook runs standalone. The executor injects
`df`/`pd`/`np` and reads back `fig`, `result_text`, `result_table`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class Recipe:
    key: str
    title: str          # e.g. "Covariance Matrix"
    concept: str        # short concept tag
    code: str           # self-contained snippet assigning `fig`
    columns: list[str] = field(default_factory=list)


@dataclass
class ExploreResult:
    recipe: Recipe
    status: str          # ok | error | skipped
    fig: Any = None      # matplotlib Figure
    table: Any = None    # pandas DataFrame
    text: str = ""
    error: str | None = None


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def categorical_columns(df: pd.DataFrame, schema: dict[str, str] | None) -> list[str]:
    if schema:
        return [c for c, t in schema.items() if t == "categorical" and c in df.columns]
    return [c for c in df.columns if df[c].dtype == object and df[c].nunique() <= 30]


# --- Recipe builders ---------------------------------------------------------
# Each takes (df, schema, target) and returns 0+ specialized Recipes.

def _b_summary(df, schema, target) -> list[Recipe]:
    return [Recipe("summary", "Summary Statistics", "describe", _CODE_SUMMARY)]


def _b_covariance(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2:
        return []
    return [Recipe("covariance", "Covariance Matrix", "linear algebra", _CODE_COVARIANCE)]


def _b_eigen(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 3:
        return []
    return [Recipe("eigen", "Eigenvectors & Eigenvalues", "linear algebra", _CODE_EIGEN)]


def _b_pca(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 3:
        return []
    return [Recipe("pca", "Dimensionality Reduction (PCA)", "unsupervised", _CODE_PCA)]


def _b_correlation(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2:
        return []
    return [Recipe("correlation", "Correlation Heatmap", "linear algebra", _CODE_CORR)]


def _b_distance(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2:
        return []
    return [Recipe("distance", "Distance & Scaling Measures", "distance", _CODE_DISTANCE)]


def _b_kmeans(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2 or len(df) < 10:
        return []
    return [Recipe("kmeans", "K-means Clustering", "unsupervised", _CODE_KMEANS)]


def _b_gmm(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2 or len(df) < 10:
        return []
    return [Recipe("gmm", "Gaussian Mixture Models", "unsupervised", _CODE_GMM)]


def _b_hierarchical(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2 or len(df) < 4:
        return []
    return [Recipe("hierarchical", "Hierarchical Clustering", "unsupervised", _CODE_DENDRO)]


def _b_dbscan(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 2 or len(df) < 10:
        return []
    return [Recipe("dbscan", "DBSCAN", "unsupervised", _CODE_DBSCAN)]


def _b_distribution(df, schema, target) -> list[Recipe]:
    cols = numeric_columns(df)
    if not cols:
        return []
    col = target if (target in cols) else cols[0]
    return [Recipe("distribution", f"Distribution Fit — {col}", "distributions",
                   _CODE_DISTRIBUTION.replace("__COL__", repr(col)), [col])]


def _b_power_law(df, schema, target) -> list[Recipe]:
    cols = [c for c in numeric_columns(df) if (df[c].dropna() > 0).all() and df[c].nunique() > 5]
    if not cols:
        return []
    # Prefer the most heavy-tailed (highest positive skew) column — that's where
    # a power-law is actually meaningful (e.g. raw volume over log-volume).
    col = max(cols, key=lambda c: float(df[c].dropna().skew()))
    return [Recipe("power_law", f"Power-Law Distribution — {col}", "distributions",
                   _CODE_POWERLAW.replace("__COL__", repr(col)), [col])]


def _b_kl(df, schema, target) -> list[Recipe]:
    cols = numeric_columns(df)
    if not cols:
        return []
    col = cols[0]
    return [Recipe("kl_divergence", f"KL Divergence vs Normal — {col}", "distributions",
                   _CODE_KL.replace("__COL__", repr(col)), [col])]


def _b_bernoulli(df, schema, target) -> list[Recipe]:
    binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if not binary:
        return []
    col = target if target in binary else binary[0]
    return [Recipe("bernoulli", f"Bernoulli / Binomial — {col}", "distributions",
                   _CODE_BERNOULLI.replace("__COL__", repr(col)), [col])]


def _b_graph(df, schema, target) -> list[Recipe]:
    if len(numeric_columns(df)) < 3:
        return []
    return [Recipe("feature_graph", "Graph Theory — Feature Correlation Graph",
                   "graph theory", _CODE_GRAPH)]


_BUILDERS: list[Callable] = [
    _b_summary, _b_covariance, _b_correlation, _b_eigen, _b_pca, _b_distance,
    _b_kmeans, _b_gmm, _b_hierarchical, _b_dbscan, _b_distribution,
    _b_power_law, _b_kl, _b_bernoulli, _b_graph,
]


def suggest_recipes(df: pd.DataFrame, schema: dict[str, str] | None = None,
                    target: str | None = None) -> list[Recipe]:
    """Return the applicable concept recipes for this dataset, in display order."""
    out: list[Recipe] = []
    for builder in _BUILDERS:
        try:
            out.extend(builder(df, schema or {}, target))
        except Exception:
            continue
    return out


def recipe_by_key(df, schema, target, key: str) -> Recipe | None:
    for r in suggest_recipes(df, schema, target):
        if r.key == key:
            return r
    return None


# --- Execution ---------------------------------------------------------------

def run_recipe(df: pd.DataFrame, recipe: Recipe) -> ExploreResult:
    """Execute a recipe's code against `df` and capture its figure/table/text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns: dict[str, Any] = {
        "df": df, "pd": pd, "np": np, "plt": plt,
        "fig": None, "result_text": "", "result_table": None,
    }
    try:
        exec(recipe.code, ns)  # noqa: S102 — local power-user tool, user's own data
        return ExploreResult(
            recipe=recipe, status="ok",
            fig=ns.get("fig"), table=ns.get("result_table"),
            text=str(ns.get("result_text") or ""),
        )
    except Exception as exc:
        return ExploreResult(recipe=recipe, status="error",
                             error=f"{type(exc).__name__}: {exc}")


# --- Notebook export ---------------------------------------------------------

def _setup_code(project) -> str:
    return (
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "try:\n    import seaborn as sns\nexcept Exception:\n    sns = None\n\n"
        f"# Project: {project.meta.get('name', project.id)}\n"
        "df = pd.read_parquet('data/working.parquet')\n"
        f"target = {project.meta.get('target_column')!r}\n"
        "df.head()"
    )


def export_notebook(project, recipes: list[Recipe] | None = None,
                    *, path: Path | None = None) -> Path:
    """Write a standalone .ipynb of the exploration. Returns the path."""
    import nbformat as nbf

    if recipes is None:
        df = project.get_working_data()
        recipes = suggest_recipes(df, project.meta.get("schema"),
                                  project.meta.get("target_column"))

    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            f"# Dataset Exploration — {project.meta.get('name', project.id)}\n\n"
            "Generated by MIRA. Each section applies one analysis concept and the "
            "chart most useful for it."),
        nbf.v4.new_code_cell(_setup_code(project)),
    ]
    for r in recipes:
        cells.append(nbf.v4.new_markdown_cell(f"## {r.title}\n_{r.concept}_"))
        cells.append(nbf.v4.new_code_cell(r.code))
    nb["cells"] = cells

    out = path or (project.root / "exports" / "exploration.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return out


# --- Code snippets (self-contained; assume `df`) -----------------------------

_CODE_SUMMARY = """\
result_table = df.describe(include='all').transpose()
fig, ax = plt.subplots(figsize=(7, 0.4 * len(df.columns) + 1))
miss = df.isna().mean().mul(100).sort_values(ascending=True)
ax.barh(miss.index.astype(str), miss.values, color='#4C78A8')
ax.set_xlabel('% missing'); ax.set_title('Missingness by column')
fig.tight_layout()
"""

_CODE_COVARIANCE = """\
import numpy as np
num = df.select_dtypes(include=np.number).dropna()
cov = np.cov(num.values, rowvar=False)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cov, cmap='RdBu', aspect='auto')
ax.set_xticks(range(num.shape[1])); ax.set_xticklabels(num.columns, rotation=90, fontsize=8)
ax.set_yticks(range(num.shape[1])); ax.set_yticklabels(num.columns, fontsize=8)
ax.set_title('Covariance matrix')
fig.colorbar(im, ax=ax); fig.tight_layout()
"""

_CODE_CORR = """\
import numpy as np
num = df.select_dtypes(include=np.number)
corr = num.corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
ax.set_title('Correlation heatmap')
fig.colorbar(im, ax=ax); fig.tight_layout()
"""

_CODE_EIGEN = """\
import numpy as np
num = df.select_dtypes(include=np.number).dropna()
Xs = (num - num.mean()) / num.std(ddof=0)
cov = np.cov(Xs.values, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
result_text = 'Eigenvalues (desc): ' + ', '.join(f'{v:.3f}' for v in eigvals)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(range(1, len(eigvals) + 1), eigvals, 'o-'); a1.set_title('Scree (eigenvalues)')
a1.set_xlabel('component'); a1.set_ylabel('eigenvalue')
a2.bar(range(1, len(eigvals) + 1), np.cumsum(eigvals) / eigvals.sum())
a2.set_title('Cumulative explained variance'); a2.set_xlabel('components'); a2.set_ylim(0, 1)
fig.tight_layout()
"""

_CODE_PCA = """\
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
num = df.select_dtypes(include=np.number).dropna()
X = StandardScaler().fit_transform(num.values)
p = PCA(n_components=2).fit(X)
Z = p.transform(X)
result_text = 'Explained variance: ' + ', '.join(f'{r:.1%}' for r in p.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.5, c=Z[:, 0], cmap='viridis')
ax.set_xlabel(f'PC1 ({p.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({p.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA 2-D projection'); fig.tight_layout()
"""

_CODE_DISTANCE = """\
import numpy as np
from scipy.spatial.distance import pdist, squareform
num = df.select_dtypes(include=np.number).dropna().head(60)
raw = squareform(pdist(num.values))
Xs = (num - num.mean()) / num.std(ddof=0).replace(0, 1)
scaled = squareform(pdist(Xs.values))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.imshow(raw, cmap='magma'); a1.set_title('Pairwise distance (raw)')
a2.imshow(scaled, cmap='magma'); a2.set_title('Pairwise distance (standardized)')
fig.tight_layout()
"""

_CODE_KMEANS = """\
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
num = df.select_dtypes(include=np.number).dropna().head(3000)
X = StandardScaler().fit_transform(num.values)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in range(1, 8)]
k = 3
labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
Z = PCA(n_components=2).fit_transform(X)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(range(1, 8), inertias, 'o-'); a1.set_title('Elbow (inertia)'); a1.set_xlabel('k')
a2.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
a2.set_title(f'K-means (k={k}) on PCA'); fig.tight_layout()
"""

_CODE_GMM = """\
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
num = df.select_dtypes(include=np.number).dropna().head(3000)
X = StandardScaler().fit_transform(num.values)
Z = PCA(n_components=2).fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=0).fit(Z)
labels = gmm.predict(Z)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='X', c='black', s=120)
ax.set_title('Gaussian Mixture (3 components) on PCA'); fig.tight_layout()
"""

_CODE_DENDRO = """\
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
num = df.select_dtypes(include=np.number).dropna().head(40)
X = StandardScaler().fit_transform(num.values)
Z = linkage(X, method='ward')
fig, ax = plt.subplots(figsize=(9, 4))
dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * Z[:, 2].max())
ax.set_title('Hierarchical clustering (Ward) dendrogram'); fig.tight_layout()
"""

_CODE_DBSCAN = """\
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
num = df.select_dtypes(include=np.number).dropna().head(3000)
X = StandardScaler().fit_transform(num.values)
Z = PCA(n_components=2).fit_transform(X)
labels = DBSCAN(eps=0.8, min_samples=8).fit_predict(Z)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
result_text = f'DBSCAN found {n_clusters} clusters; {int((labels==-1).sum())} noise points'
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
ax.set_title('DBSCAN density clusters on PCA'); fig.tight_layout()
"""

_CODE_DISTRIBUTION = """\
import numpy as np
from scipy import stats
col = __COL__
x = pd.to_numeric(df[col], errors='coerce').dropna()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(x, bins=40, density=True, alpha=0.6, color='#4C78A8')
xs = np.linspace(x.min(), x.max(), 200)
ax.plot(xs, stats.norm(x.mean(), x.std()).pdf(xs), 'r-', label='Normal fit')
try:
    kde = stats.gaussian_kde(x); ax.plot(xs, kde(xs), 'g--', label='KDE')
except Exception:
    pass
ax.set_title(f'Distribution of {col}'); ax.legend(); fig.tight_layout()
result_text = f'mean={x.mean():.4g} std={x.std():.4g} skew={stats.skew(x):.4g} kurtosis={stats.kurtosis(x):.4g}'
"""

_CODE_POWERLAW = """\
import numpy as np
col = __COL__
x = pd.to_numeric(df[col], errors='coerce').dropna()
x = x[x > 0].sort_values(ascending=False).values
ranks = np.arange(1, len(x) + 1)
slope, intercept = np.polyfit(np.log(ranks), np.log(x), 1)
result_text = f'Rank-frequency log-log slope = {slope:.3f} (steeper negative => heavier tail)'
fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(ranks, x, '.', alpha=0.6)
ax.loglog(ranks, np.exp(intercept) * ranks ** slope, 'r-', label=f'slope={slope:.2f}')
ax.set_xlabel('rank'); ax.set_ylabel(col); ax.set_title('Power-law (rank-frequency)')
ax.legend(); fig.tight_layout()
"""

_CODE_KL = """\
import numpy as np
from scipy import stats
col = __COL__
x = pd.to_numeric(df[col], errors='coerce').dropna()
bins = np.linspace(x.min(), x.max(), 30)
p, _ = np.histogram(x, bins=bins, density=True)
centers = (bins[:-1] + bins[1:]) / 2
q = stats.norm(x.mean(), x.std()).pdf(centers)
p = p + 1e-9; q = q + 1e-9
p = p / p.sum(); q = q / q.sum()
kl = float(np.sum(p * np.log(p / q)))
result_text = f'KL(data || Normal) = {kl:.4f} nats'
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(centers, p, width=(bins[1]-bins[0]) * 0.9, alpha=0.5, label='data')
ax.plot(centers, q, 'r-', label='Normal reference')
ax.set_title(f'KL divergence of {col} vs Normal = {kl:.3f}'); ax.legend(); fig.tight_layout()
"""

_CODE_BERNOULLI = """\
import numpy as np
from scipy import stats
col = __COL__
s = df[col].dropna()
vals = sorted(s.unique())
mapping = {vals[0]: 0, vals[1]: 1}
b = s.map(mapping)
p = float(b.mean())
n = len(b)
result_text = f'Bernoulli p(success={vals[1]!r}) = {p:.4f}  over n={n}'
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.bar([str(vals[0]), str(vals[1])], [1 - p, p], color=['#E45756', '#4C78A8'])
a1.set_title(f'Bernoulli estimate (p={p:.3f})')
ks = np.arange(0, n + 1)
a2.plot(ks, stats.binom(n, p).pmf(ks), '.-')
a2.set_title(f'Binomial(n={n}, p={p:.3f}) PMF'); a2.set_xlabel('successes')
fig.tight_layout()
"""

_CODE_GRAPH = """\
import numpy as np
num = df.select_dtypes(include=np.number)
corr = num.corr().fillna(0).values
labels = list(num.columns)
thr = 0.5
A = (np.abs(corr) >= thr).astype(int)
np.fill_diagonal(A, 0)
n = len(labels)
ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos = np.column_stack([np.cos(ang), np.sin(ang)])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
im = a1.imshow(A, cmap='Greys'); a1.set_title('Adjacency matrix (|corr| >= 0.5)')
a1.set_xticks(range(n)); a1.set_xticklabels(labels, rotation=90, fontsize=7)
a1.set_yticks(range(n)); a1.set_yticklabels(labels, fontsize=7)
for i in range(n):
    for j in range(i + 1, n):
        if A[i, j]:
            a2.plot(pos[[i, j], 0], pos[[i, j], 1], '-', color='#999', lw=1, alpha=0.6)
a2.scatter(pos[:, 0], pos[:, 1], s=300, c='#4C78A8', zorder=3)
for i, lab in enumerate(labels):
    a2.text(pos[i, 0], pos[i, 1], lab, ha='center', va='center', fontsize=7, zorder=4)
a2.set_title('Feature correlation graph'); a2.axis('off')
fig.tight_layout()
"""
