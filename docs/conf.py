import datetime
import os
import shutil

import audeer


# Project -----------------------------------------------------------------

project = 'auglib'
copyright = f'2019-{datetime.date.today().year} audEERING GmbH'
author = 'Johannes Wagner, Hagen Wierstorf'
version = audeer.git_repo_version()
title = '{} Documentation'.format(project)


# General -----------------------------------------------------------------

master_doc = 'index'
source_suffix = '.rst'
exclude_patterns = [
    'api-src',
    'build',
    'tests',
    'Thumbs.db',
    '.DS_Store',
]
pygments_style = None
extensions = [
    'jupyter_sphinx',  # executing code blocks
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',  # for "copy to clipboard" buttons
]
templates_path = ['_templates']

# Disable auto-generation of TOC entries in the API
# https://github.com/sphinx-doc/sphinx/issues/6316
toc_object_entries = False

copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = False  # disable docstring inheritance
typehints_fully_qualified = True  # show fully qualified class names

intersphinx_mapping = {
    'audformat': ('https://audeering.github.io/audformat/', None),
    'audobject': ('https://audeering.github.io/audobject', None),
    'audresample': ('https://audeering.github.io/audresample/', None),
    'audtorch': ('https://audeering.github.io/audtorch/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'parselmouth': ('https://parselmouth.readthedocs.io/en/stable', None),
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
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
    'https://www.sphinx-doc.org',
]


# Copy API (sub-)module RST files to docs/api/ folder ---------------------
audeer.rmdir('api')
audeer.mkdir('api')
api_src_files = audeer.list_file_names('api-src')
api_dst_files = [
    audeer.path('api', os.path.basename(src_file))
    for src_file in api_src_files
]
for src_file, dst_file in zip(api_src_files, api_dst_files):
    shutil.copyfile(src_file, dst_file)
