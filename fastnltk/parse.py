"""
fastnltk.parse — Drop-in replacement for nltk.parse.

Rust-accelerated CFG + EarleyChartParser for context-free grammar parsing.
Other parsers (chart, dependency, PCFG, etc.) fall back to NLTK.
"""

import nltk.parse as _nltk_parse
from nltk.parse import *  # noqa: F403

from fastnltk._rust import CFG as _RustCFG
from fastnltk._rust import EarleyChartParser as _RustEarleyChartParser


class CFG:
    """Context-free grammar — Rust-accelerated."""
    def __init__(self, start, productions):
        self._impl = _RustCFG(start, productions)

    @classmethod
    def from_string(cls, grammar_str):
        return cls.__new__(cls)._from_impl(_RustCFG.from_string(grammar_str))

    @classmethod
    def _from_impl(cls, impl):
        inst = cls.__new__(cls)
        inst._impl = impl
        return inst

    def start(self):
        return self._impl.start()

    def productions(self):
        return self._impl.productions()

    def nonterminals(self):
        return self._impl.nonterminals()

    def __len__(self):
        return self._impl.__len__()

    def __str__(self):
        return str(self._impl)


class EarleyChartParser:
    """Earley chart parser — Rust-accelerated."""
    def __init__(self):
        self._impl = _RustEarleyChartParser()

    def parse(self, grammar, tokens):
        return self._impl.parse(grammar._impl if hasattr(grammar, "_impl") else grammar, tokens)

    def parse_sents(self, grammar, sentences):
        return [self.parse(grammar, s) for s in sentences]
