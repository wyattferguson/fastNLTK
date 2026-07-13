"""
fastnltk.parse — Pure Python shim wrapping nltk.parse.

Parsing algorithms are complex, rarely performance-critical for
most NLTK users. Keeping as pure Python shim.
"""

import nltk.parse as _nltk_parse
from nltk.parse import *  # noqa: F401, F403
