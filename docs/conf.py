# Sphinx configuration for linumpy-manual-align.
# Mirrors the FIrgolitsch/linumpy `dev` branch docs/conf.py with 8
# project-specific adaptations (see .planning/phases/11-sphinx-documentation).

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "linumpy-manual-align"
author = "The LINUM developers"
copyright = f"{datetime.now().year}, LINUM"

try:
    from importlib.metadata import version as _get_version
    release = _get_version("linumpy-manual-align")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon",
    "sphinx.ext.viewcode", "sphinx.ext.intersphinx", "sphinx.ext.mathjax",
    "autoapi.extension", "sphinxarg.ext", "myst_parser", "sphinx_design",
    "sphinxcontrib.mermaid", "sphinx_copybutton", "notfound.extension",
    "sphinx_sitemap", "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = [
    "amsmath", "colon_fence", "deflist", "dollarmath", "fieldlist",
    "linkify", "substitution", "tasklist",
]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 4

# Mermaid — startOnLoad=False is critical (sphinxcontrib-mermaid calls
# mermaid.run() itself after wiring d3 zoom; startOnLoad=True would auto-render
# before the wrapper attaches, breaking zoom/fullscreen buttons).
mermaid_d3_zoom = True
mermaid_fullscreen = True
mermaid_fullscreen_button = "⛶"
mermaid_height = "640px"
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_init_config = {
    "startOnLoad": False, "securityLevel": "loose",
    "flowchart": {"htmlLabels": True, "curve": "basis", "useMaxWidth": True},
    "themeVariables": {"fontSize": "16px"},
}

# AutoAPI — src layout: point at src/linumpy_manual_align specifically,
# NOT at src/ (the parent) which would produce an empty docs/api/ directory.
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "src" / "linumpy_manual_align")]
autoapi_root = "api"
autoapi_options = [
    "members", "undoc-members", "show-inheritance",
    "show-module-summary", "imported-members",
]
autoapi_ignore = ["*/tests/*"]
autoapi_keep_files = True
autoapi_add_toctree_entry = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon — NumPy docstrings (matches our ruff convention = "numpy")
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
napoleon_use_ivar = True  # avoids duplicate-object-description warnings

# Intersphinx — keep our deps + add linumpy (D-04; verified live inventory)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "SimpleITK": ("https://simpleitk.readthedocs.io/en/master/", None),
    "linumpy": ("https://linumpy.readthedocs.io/en/latest/", None),
}
nitpicky = False  # keep off — autoapi extracts informal numpy-style type names

html_theme = "pydata_sphinx_theme"
html_title = "linumpy-manual-align"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/linum-uqam/linumpy-manual-align",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {"name": "GitHub",
         "url": "https://github.com/linum-uqam/linumpy-manual-align",
         "icon": "fa-brands fa-github"},
    ],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    # page keys match OUR page names (cli, architecture land in wave 3)
    "secondary_sidebar_items": {
        "**": ["page-toc"],
        "index": [],
        "cli": [],
        "architecture": [],
    },
}

html_context = {
    "github_user": "linum-uqam",
    "github_repo": "linumpy-manual-align",
    "github_version": "main",
    "doc_path": "docs",
}

suppress_warnings = [
    "autoapi.python_import_resolution", "ref.python", "misc.highlighting_failure",
]

# sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# sphinx-notfound-page
notfound_context = {
    "title": "Page not found",
    "body": ("<h1>Page not found</h1><p>Sorry, we couldn't find that page. "
             "Try the <a href='/'>documentation home</a> or use the search box above.</p>"),
}
notfound_urls_prefix = "/"

# sphinx-sitemap + opengraph
html_baseurl = "https://linumpy-manual-align.readthedocs.io/en/latest/"
sitemap_url_scheme = "{link}"
ogp_site_url = html_baseurl
ogp_site_name = "linumpy-manual-align documentation"
# ogp_image — omitted initially (deferred-idea: no logo asset yet)
ogp_use_first_image = True
ogp_enable_meta_description = True
