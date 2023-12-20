import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spc_charts'
copyright = '2023, Julian Wright'
author = 'Julian Wright'
release = '0.0.1'

extensions = ['sphinx.ext.autodoc']
templates_path = ['_templates']
exclude_patterns = []

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

