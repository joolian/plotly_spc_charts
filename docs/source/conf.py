import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'plotly_spc_charts'
copyright = '2023, Julian Wright'
author = 'Julian Wright'
release = '0.0.1'

extensions = [
        'autoclasstoc',
        'sphinx.ext.autodoc'
]
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autoclass_content = 'both'

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'exclude-members': '__weakref__',
    'member-order': 'bysource'
}

