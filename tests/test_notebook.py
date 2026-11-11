"""Tests for the notebook engine (core/notebook.py)."""

from __future__ import annotations

import pandas as pd

from core import notebook as nb


def _df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})


# --- kernel execution -------------------------------------------------------- #

def test_trailing_expression_is_captured():
    k = nb.Kernel(_df())
    res = k.execute("df['a'].sum()")
    assert res.ok
    assert res.value == 10
    assert res.error is None


def test_stdout_is_captured():
    k = nb.Kernel(_df())
    res = k.execute("print('hello', 2 + 2)")
    assert res.ok
    assert "hello 4" in res.stdout
    assert res.value is None  # a print statement is not a trailing expression


def test_dataframe_value_roundtrips():
    k = nb.Kernel(_df())
    res = k.execute("df.head(2)")
    assert res.ok
    assert isinstance(res.value, pd.DataFrame)
    assert len(res.value) == 2


def test_namespace_persists_across_cells():
    k = nb.Kernel(_df())
    assert k.execute("x = df['b'].mean()").ok
    res = k.execute("x")
    assert res.value == 25.0


def test_error_is_captured_not_raised():
    k = nb.Kernel(_df())
    res = k.execute("df['nonexistent_column']")
    assert not res.ok
    assert res.error is not None
    assert "KeyError" in res.error


def test_syntax_error_is_captured():
    k = nb.Kernel(_df())
    res = k.execute("for x in :")
    assert not res.ok
    assert "SyntaxError" in (res.error or "")


def test_empty_code_is_noop():
    k = nb.Kernel(_df())
    res = k.execute("   \n  ")
    assert res.ok
    assert not res.has_output


def test_matplotlib_figure_is_captured():
    k = nb.Kernel(_df())
    res = k.execute("plt.figure(); plt.plot(df['a'], df['b'])")
    assert res.ok
    assert len(res.figures) == 1


def test_reset_clears_user_vars_but_keeps_df():
    k = nb.Kernel(_df())
    k.execute("secret = 42")
    k.reset(_df())
    res = k.execute("secret")
    assert not res.ok  # NameError after reset
    assert k.execute("len(df)").value == 4


# --- persistence ------------------------------------------------------------- #

class _FakeProject:
    def __init__(self, root):
        self.root = root
        self.id = "test1234"


def test_save_and_load_cells(tmp_path):
    proj = _FakeProject(tmp_path)
    cells = [nb.new_cell("df.head()"), nb.new_cell("df.describe()")]
    path = nb.save_cells(proj, cells)
    assert path.exists()

    loaded = nb.load_cells(proj)
    assert [c.source for c in loaded] == ["df.head()", "df.describe()"]
    assert [c.id for c in loaded] == [c.id for c in cells]


def test_load_missing_returns_default(tmp_path):
    proj = _FakeProject(tmp_path)
    loaded = nb.load_cells(proj)
    assert len(loaded) >= 1  # friendly starter notebook


def test_load_corrupt_returns_default(tmp_path):
    proj = _FakeProject(tmp_path)
    nb.notebook_path(proj).write_text("{not json", encoding="utf-8")
    loaded = nb.load_cells(proj)
    assert len(loaded) >= 1


# --- AI helpers -------------------------------------------------------------- #

def test_extract_code_from_fenced_block():
    reply = "Sure!\n```python\ndf.groupby('a').mean()\n```\nHope that helps."
    assert nb.extract_code(reply) == "df.groupby('a').mean()"


def test_extract_code_without_language_tag():
    reply = "```\ndf.head()\n```"
    assert nb.extract_code(reply) == "df.head()"


def test_extract_code_falls_back_to_raw():
    assert nb.extract_code("df.head()") == "df.head()"


def test_dataset_context_lists_columns():
    ctx = nb.dataset_context(_df(), target="b")
    assert "4 rows x 2 columns" in ctx
    assert "a: int64" in ctx
    assert "Target column: b" in ctx


def test_build_edit_messages_new_vs_edit():
    new_msg = nb.build_edit_messages("plot it", "", "CTX", is_new=True)
    assert "Write a new notebook cell" in new_msg[0]["content"]

    edit_msg = nb.build_edit_messages("sort it", "df.head()", "CTX")
    assert "current cell" in edit_msg[0]["content"]
    assert "df.head()" in edit_msg[0]["content"]
