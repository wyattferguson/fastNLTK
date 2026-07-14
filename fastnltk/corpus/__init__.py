"""
fastnltk.corpus — Drop-in replacement for nltk.corpus.

Pure Python shim — corpus reading is I/O bound, Rust offers no benefit.
"""

from nltk.corpus import *  # noqa: F403
