"""The evals page renders, and keeps rendering when a section has no data.

These are cheap guards against the two ways this page has actually broken:
a placeholder left unreplaced because one of two call sites was missed, and a
JavaScript syntax error inside a template string, which shows up at runtime as
a blank section rather than an exception anywhere near the mistake.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report


def page_script(html: str) -> str:
    return max(re.findall(r"<script>(.*?)</script>", html, re.S), key=len)


def test_no_placeholder_survives_rendering():
    html = report.render_evals([])
    left = re.findall(r"__[A-Z_]+__", html)
    assert left == [], left


def test_the_page_renders_with_no_label_sets(monkeypatch):
    """The hand-labels section hides itself rather than rendering empty."""
    monkeypatch.setattr(report, "label_sets", lambda: [])
    html = report.render_evals([])
    assert "const LABELS = [];" in html
    assert 'id="labels-section" hidden' in html


def test_the_page_embeds_the_label_sets_it_is_given(monkeypatch):
    monkeypatch.setattr(report, "label_sets", lambda: [{"set_id": "s1", "n_items": 3}])
    html = report.render_evals([])
    assert '"set_id": "s1"' in html


def test_label_sets_skips_a_result_file_it_cannot_parse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "result_good.json").write_text('{"set_id": "good"}')
    (tmp_path / "labels" / "result_bad.json").write_text("{not json")
    assert [s["set_id"] for s in report.label_sets()] == ["good"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_the_pages_javascript_parses():
    """A syntax error in a template literal is reported dozens of lines away
    from its cause, so it is worth catching here rather than in a browser."""
    for html in (report.render_evals([]), report.render_evals(report.eval_rows([]))):
        done = subprocess.run(["node", "--check", "-"], input=page_script(html),
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
