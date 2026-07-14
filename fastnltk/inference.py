"""
fastnltk.inference — Drop-in replacement for nltk.inference.

Theorem prover interfaces (Prover9, Mace, Tableau, Resolution).
Full Rust-accelerated stack: TableauProver, ResolutionProver,
DiscourseThread, DefaultReasoner, ClosedWorldReasoner.
"""

from fastnltk._rust import TableauProver, ResolutionProver, ProverResult
from fastnltk._rust import DiscourseThread, DefaultReasoner, DefaultRule, ClosedWorldReasoner

# Re-export NLTK names for API compatibility
from nltk.inference import (  # noqa: F401
    Prover9,
    Mace,
    ProverCommand,
    Prover,
    get_prover,
)
