"""
tests/test_explore.py

Phase 10: notebook-style exploration. Verifies recipe suggestion/gating, that
every applicable recipe's self-contained code executes and yields a figure, and
that the .ipynb export is valid. Headless matplotlib (Agg).
Run with: pytest tests/test_explore.py -q
"""

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def temp_projects_dir(monkeypatch, tmp_path):
    (tmp_path / "projects").mkdir()
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _rich_df(n=300):
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, n)
    return pd.DataFrame({
        "a": a,
        "b": a * 0.8 + rng.normal(0, 0.3, n),   # correlated
        "c": rng.exponential(2.0, n) + 0.1,     # positive, heavy-ish tail
        "d": rng.normal(5, 2, n),
        "flag": rng.randint(0, 2, n),           # binary -> Bernoulli
        "y": a * 2 + rng.normal(0, 0.5, n),     # regression target
    })


def test_suggest_includes_concept_recipes(temp_projects_dir):
    from core import explore

    recipes = {r.key for r in explore.suggest_recipes(_rich_df(), None, "y")}
    # The lecture concepts should all be offered for a rich numeric dataset.
    for key in ["covariance", "eigen", "pca", "correlation", "distance",
                "kmeans", "gmm", "hierarchical", "dbscan", "distribution",
                "power_law", "kl_divergence", "bernoulli", "feature_graph"]:
        assert key in recipes, f"missing recipe: {key}"


def test_gating_excludes_when_too_few_numeric(temp_projects_dir):
    from core import explore

    df = pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0]})
    keys = {r.key for r in explore.suggest_recipes(df, None, None)}
    # 1 numeric column -> no covariance/pca/clustering
    assert "covariance" not in keys
    assert "pca" not in keys
    assert "summary" in keys  # always applicable


def test_every_suggested_recipe_runs_and_makes_a_figure(temp_projects_dir):
    from core import explore

    df = _rich_df()
    recipes = explore.suggest_recipes(df, None, "y")
    assert len(recipes) >= 12
    for r in recipes:
        res = explore.run_recipe(df, r)
        assert res.status == "ok", f"{r.key} failed: {res.error}"
        assert res.fig is not None, f"{r.key} produced no figure"


def test_recipes_emit_expected_text(temp_projects_dir):
    from core import explore

    df = _rich_df()
    by_key = {r.key: r for r in explore.suggest_recipes(df, None, "y")}
    assert "Explained variance" in explore.run_recipe(df, by_key["pca"]).text
    assert "KL(" in explore.run_recipe(df, by_key["kl_divergence"]).text
    assert "slope" in explore.run_recipe(df, by_key["power_law"]).text
    assert "Bernoulli p" in explore.run_recipe(df, by_key["bernoulli"]).text


def test_export_notebook_is_valid(temp_projects_dir):
    import nbformat

    from core import explore
    from core.project import Project

    p = Project.create(name="Explore NB")
    p.set_raw_data(_rich_df())
    p.meta["target_column"] = "y"
    p.save_meta()

    path = explore.export_notebook(p)
    assert path.exists()
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    # setup cell + a markdown/code pair per recipe
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 10


def test_cli_data_explore_notebook(temp_projects_dir, capsys):
    from core.project import Project
    from mira_cli import main

    p = Project.create(name="Explore CLI")
    p.set_raw_data(_rich_df())
    p.meta["target_column"] = "y"
    p.save_meta()

    rc = main(["data", "explore", "--project-id", p.id, "--run", "pca", "--notebook"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Suggested analyses" in out
    assert "Dimensionality Reduction (PCA)" in out
    assert "exploration.ipynb" in out
    assert (p.root / "exports" / "exploration.ipynb").exists()
