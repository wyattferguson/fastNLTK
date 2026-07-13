"""fastnltk.translate — Drop-in replacement for nltk.translate."""

_rust_available = False
try:
    from fastnltk._rust import bleu_score as _rust_bleu, corpus_bleu as _rust_corpus_bleu
    _rust_available = True
except ImportError:
    pass

import nltk.translate as _nltk_translate
from nltk.translate import *  # noqa: F401, F403

__all__ = ["bleu_score", "corpus_bleu"]

def bleu_score(candidate, reference, max_n=4):
    if _rust_available:
        return _rust_bleu(candidate, reference, max_n)
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([reference], candidate, max_n=max_n)

def corpus_bleu(candidates, references, max_n=4):
    if _rust_available:
        return _rust_corpus_bleu(candidates, references, max_n)
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
    return nltk_corpus_bleu([[r] for r in references], candidates, max_n=max_n)
