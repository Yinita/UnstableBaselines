import os, sys
sys.path.insert(0, os.path.abspath('../unstable'))


# -- Project information -----------------------------------------------------
project = "Unstable Baselines"
copyright = "2025, Leon Guertler"
author = "Leon Guertler, Tim Grams, Liu Zichen, Bobby Cheng"

# -- General configuration ---------------------------------------------------
master_doc = 'index'
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]
templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = [
    "torch", "ray", "vllm", "textarena", "pandas", "numpy",
    "rich", "trueskill",  # add others your code imports
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
autosummary_generate = True
# html_static_path = ['_static']