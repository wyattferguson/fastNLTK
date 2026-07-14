"""
fastnltk.parse — Drop-in replacement for nltk.parse.

Pure Python shim — parsing algorithms are complex, rarely performance-critical
for most NLTK users. Keeping as pure Python wrapper.
"""

from nltk.parse import *  # noqa: F403
