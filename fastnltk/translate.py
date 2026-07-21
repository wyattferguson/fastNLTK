"""fastnltk.translate — Drop-in replacement for nltk.translate."""

from nltk.translate import *  # noqa: F403

from fastnltk._rust import bleu_score as _rust_bleu
from fastnltk._rust import corpus_bleu as _rust_corpus_bleu

__all__ = ["bleu_score", "corpus_bleu"]


def bleu_score(candidate: list[str], reference: list[list[str]], max_n: int = 4) -> float:
    return _rust_bleu(candidate, reference, max_n)


def corpus_bleu(
    candidates: list[list[str]], references: list[list[list[str]]], max_n: int = 4
) -> float:
    return _rust_corpus_bleu(candidates, references, max_n)
