from subprocess import check_output


# Project -----------------------------------------------------------------

project = 'auglib'
copyright = '2019-2020 audEERING GmbH'
author = ('Johannes Wagner')
# The x.y.z version read from tags
try:
    version = check_output(['git', 'describe', '--tags', '--always'])
    version = version.decode().strip()
except Exception:
    version = '<unknown>'
title = '{} Documentation'.format(project)


# General -----------------------------------------------------------------

master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = None
extensions = [
    'jupyter_sphinx',  # executing code blocks
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',  # for "copy to clipboard" buttons
]

copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = False  # disable docstring inheritance
typehints_fully_qualified = True  # show fully qualified class names

intersphinx_mapping = {
    'audformat': ('https://audeering.github.io/audformat/', None),
    'audobject': ('https://audeering.github.io/audobject', None),
    'audresample': ('https://audeering.github.io/audresample/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}


# HTML --------------------------------------------------------------------

html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
}
html_title = title


# Linkcheck ---------------------------------------------------------------

linkcheck_ignore = [
    r'https://sail.usc.edu/',
    'https://sphinx-doc.org/',
]
