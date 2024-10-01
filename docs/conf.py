# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'dltDFLISR'
copyright = 'Battelle Memorial Institute, 2024'
author = 'Fernando Bereta dos Reis'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     'myst_parser',
#     'sphinxcontrib.plantuml',
#     'sphinx.ext.todo',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.napoleon',
#     'sphinx_rtd_theme',
#     'sphinx.ext.mathjax',
#     ]

extensions = [
    'myst_parser',               # MyST markdown support
    'sphinx.ext.autodoc',       # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',       # Google style and NumPy style docstring support
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.mathjax',        # LaTeX support for math equations
    'sphinx.ext.todo',           # Support for TODOs in documentation
    'sphinxcontrib.bibtex',
    'sphinxcontrib.plantuml',    # PlantUML support for diagrams
    'sphinx_rtd_theme',          # Read the Docs theme
]

# # Enable MyST markdown features
# myst_enable_extensions = [
#     "amsmath",  # For LaTeX math support
#     "dollarmath"  # For $ and $$ delimiters for inline and block math
# ]

# # Ensure that mathjax is set up for LaTeX rendering
# mathjax_config = {
#     'tex2jax': {
#         'inlineMath': [['$', '$'], ['\\(', '\\)']],
#         'displayMath': [['$$', '$$'], ['\\[', '\\]']],
#     }
# }

# myst_enable_extensions = [
#     # "dmath",  # Enable directive-based math support
#     "amsmath",  # Enable AMS math support
# ]

# # mathjax_config = {
# #     'tex2jax': {
# #         'inlineMath': [['$', '$'], ['\\(', '\\)']],
# #         'displayMath': [['$$', '$$'], ['\\[', '\\]']],
# #     }
# # }
# mathjax3_config = {
#     'tex': {
#         'inlineMath': [['$', '$'], ['\\(', '\\)']],
#         'displayMath': [['$$', '$$'], ['\\[', '\\]']],
#     }
# }

# Add bibliography files for citations
bibtex_bibfiles = ['files/latex/references.bib','files/latex/document.bib']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
# myst
myst_heading_anchors = 5
# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
#     'analytics_anonymize_ip': False,
#     # 'logo_only': True,
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     'vcs_pageview_mode': '',
#     # 'style_nav_header_background': 'white',
#     # Toc options
#     'collapse_navigation': False,
#     'sticky_navigation': False,
#     'navigation_depth': 5,
#     'includehidden': False,
#     'titles_only': False
# }

html_static_path = []

source_suffix = {
    '.rst': 'restructuredtext',
    ".md":'markdown',
}

if os.environ.get('READTHEDOCS') == 'True':
    rtd_path=os.environ.get('READTHEDOCS_VIRTUALENV_PATH')
    plantuml = f"java -jar {os.path.join(rtd_path, 'bin', 'plantuml.jar')}"
    # plantuml = 'http://www.plantuml.com/plantuml' # Read the Docs remote
else:
    plantuml = f'java -jar {os.path.expanduser("~/support/plantuml-mit-1.2024.5.jar")}'

# Path to graphviz executable
# graphviz_dot = "/usr/bin/dot"

