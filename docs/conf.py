import os, sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
project = "Unstable Baselines"
copyright = "2025, Leon Guertler"
author = "Leon Guertler, Tim Grams, Liu Zichen, Bobby Cheng"

# -- General configuration ---------------------------------------------------
master_doc = 'index'
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]
templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
napoleon_google_docstring = True
napoleon_numpy_docstring = False
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']