"""Smoke test: sphinx-build succeeds and key pages/content exist.

Runs the full Sphinx site build via subprocess (slow, ~30-60s) and asserts
that the landing page grid, CLI reference, auto-generated API reference,
migrated operator guide, architecture page, and the intersphinx linumpy
lede all rendered. A separate fast unit test checks the README links to
the published Read the Docs site.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
BUILD_DIR = DOCS_DIR / "_build" / "html"


@pytest.fixture(scope="module")
def docs_build() -> None:
    """Build the Sphinx site once per module run (slow ~30-60s)."""
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(DOCS_DIR), str(BUILD_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.docs_build
def test_index_has_grid_cards(docs_build: None) -> None:
    # DOCS-05: sphinx-design grid cards rendered on the landing page.
    # sphinx-design renders ``grid-item-card`` directives as ``sd-card`` HTML
    # classes, so we assert the rendered class.
    assert "sd-card" in (BUILD_DIR / "index.html").read_text()


@pytest.mark.docs_build
def test_cli_page_has_argparse_options(docs_build: None) -> None:
    # DOCS-04: sphinx-argparse rendered the CLI surface.
    assert "data_package" in (BUILD_DIR / "cli.html").read_text()


@pytest.mark.docs_build
def test_api_reference_exists(docs_build: None) -> None:
    # DOCS-03: autoapi generated the package API index.
    assert (BUILD_DIR / "api" / "linumpy_manual_align" / "index.html").exists()


@pytest.mark.docs_build
def test_guide_migrated(docs_build: None) -> None:
    # DOCS-02: the operator guide was migrated into the Sphinx site.
    assert "Data Package Layout" in (BUILD_DIR / "cli-nextflow-guide.html").read_text()


@pytest.mark.docs_build
def test_architecture_page_exists(docs_build: None) -> None:
    # DOCS-01 / D-03: the architecture toctree page rendered.
    assert (BUILD_DIR / "architecture.html").exists()


@pytest.mark.docs_build
def test_intersphinx_linumpy_referenced(docs_build: None) -> None:
    # DOCS-06: the lede's external linumpy link is present on the landing page.
    assert "linumpy" in (BUILD_DIR / "index.html").read_text()


def test_readme_links_to_rtd() -> None:
    # DOCS-07: README points users at the published Read the Docs site.
    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert "linumpy-manual-align.readthedocs.io" in readme.read_text()
