"""
fastnltk.inference — Drop-in replacement for nltk.inference.

Theorem prover interfaces (Prover9, Mace, Tableau, Resolution).
TableauProver and ResolutionProver are Rust-accelerated;
Prover9/Mace wrappers fall back to NLTK.
"""

from nltk.inference import *  # noqa: F403

_rust_available = False
try:
    from fastnltk._rust import TableauProver
    from fastnltk._rust import ResolutionProver
    from fastnltk._rust import ProverResult
    _rust_available = True
except ImportError:
    pass
