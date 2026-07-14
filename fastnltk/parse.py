"""
fastnltk.parse — Pure Python shim wrapping nltk.parse.

Parsing algorithms are complex, rarely performance-critical for
most NLTK users. Keeping as pure Python shim.
"""

from nltk.parse import *  # noqa: F403
