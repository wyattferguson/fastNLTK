"""
fastnltk.inference — Drop-in replacement for nltk.inference.

Theorem prover interfaces (Prover9, Mace, Tableau, Resolution).
Full Rust-accelerated stack: TableauProver, ResolutionProver,
DiscourseThread, DefaultReasoner, ClosedWorldReasoner.
"""


# Re-export NLTK names for API compatibility
from nltk.inference import (  # noqa: F401
    Mace,
    Prover,
    Prover9,
    ProverCommand,
    get_prover,
)
