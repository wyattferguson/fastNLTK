"""
fastnltk.inference — Drop-in replacement for nltk.inference.

Theorem prover interfaces (Prover9, Mace, Tableau, Resolution).
Full Rust-accelerated stack: TableauProver, ResolutionProver,
DiscourseThread, DefaultReasoner, ClosedWorldReasoner.
"""


# Re-export NLTK names for API compatibility
from nltk.inference import (  # noqa: F401
    DiscourseTester,
    Mace,
    MaceCommand,
    ParallelProverBuilder,
    ParallelProverBuilderCommand,
    Prover9,
    Prover9Command,
    ResolutionProver,
    ResolutionProverCommand,
    TableauProver,
    TableauProverCommand,
)
