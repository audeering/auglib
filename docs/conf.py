import datetime
import os

import toml

import audeer


config = toml.load(audeer.path("..", "pyproject.toml"))
current_dir = os.path.dirname(os.path.realpath(__file__))


# Project -----------------------------------------------------------------

project = config["project"]["name"]
author = ", ".join(author["name"] for author in config["project"]["authors"])
copyright = f"2019-{datetime.date.today().year} audEERING GmbH"
version = audeer.git_repo_version()
title = "Documentation"


# General -----------------------------------------------------------------

master_doc = "index"
source_suffix = ".rst"
exclude_patterns = [
    "api-src",
    "build",
    "tests",
    "Thumbs.db",
    ".DS_Store",
]
pygments_style = None
extensions = [
    "jupyter_sphinx",  # executing code blocks
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # support for Google-style docstrings
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",  # for "copy to clipboard" buttons
    "matplotlib.sphinxext.plot_directive",
    "sphinx_apipages",
]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = False  # disable docstring inheritance
typehints_fully_qualified = True  # show fully qualified class names

# Code executed at the beginning of each .. plot:: directive
plot_pre_code = (
    "import audeer\n"
    "import audiofile\n"
    "import matplotlib.pyplot as plt\n"
    "media_dir = audeer.mkdir("
    f"'{current_dir}', '..', 'build', 'html', 'api', 'media')\n"
    "plt.rcParams['font.size'] = 13"
)

intersphinx_mapping = {
    "audformat": ("https://audeering.github.io/audformat/", None),
    "audobject": ("https://audeering.github.io/audobject", None),
    "audresample": ("https://audeering.github.io/audresample/", None),
    "audtorch": ("https://audeering.github.io/audtorch/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "parselmouth": ("https://parselmouth.readthedocs.io/en/stable", None),
    "pedalboard": ("https://spotify.github.io/pedalboard", None),
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Matplot plot_directive settings
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["png"]

# HTML --------------------------------------------------------------------

html_theme = "sphinx_audeering_theme"
html_theme_options = {
    "display_version": True,
    "logo_only": False,
    "footer_links": False,
}
html_context = {
    "display_github": True,
}
html_title = title


# Linkcheck ---------------------------------------------------------------

linkcheck_ignore = [
    r"https://sail.usc.edu/",
    "https://www.sphinx-doc.org",
]
