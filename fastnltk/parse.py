"""
fastnltk.parse — Drop-in replacement for nltk.parse.

Rust-accelerated CFG + EarleyChartParser for context-free grammar parsing.
Other parsers (chart, dependency, PCFG, etc.) fall back to NLTK.
"""

from nltk.parse import *  # noqa: F403

from fastnltk._rust import CFG as _RustCFG
from fastnltk._rust import EarleyChartParser as _RustEarleyChartParser


class CFG:
    """Context-free grammar — Rust-accelerated."""

    def __init__(self, start, productions):
        self._impl = _RustCFG(start, productions)

    @classmethod
    def from_string(cls, grammar_str: str) -> any:
        return cls.__new__(cls)._from_impl(_RustCFG.from_string(grammar_str))

    @classmethod
    def _from_impl(cls, impl: any) -> any:
        inst = cls.__new__(cls)
        inst._impl = impl
        return inst

    def start(self) -> any:
        return self._impl.start()

    def productions(self) -> list[any]:
        return self._impl.productions()

    def nonterminals(self) -> list[any]:
        return self._impl.nonterminals()

    def __len__(self):
        return self._impl.__len__()

    def __str__(self):
        return str(self._impl)


class EarleyChartParser:
    """Earley chart parser — Rust-accelerated."""

    def __init__(self, grammar=None, trace=0):
        self._impl = _RustEarleyChartParser()
        if grammar is not None:
            self._grammar = grammar._impl if hasattr(grammar, "_impl") else grammar
        else:
            self._grammar = None

    def parse(self, tokens_or_grammar, tokens_or_none=None, grammar=None):
        # Backward compat: old signature was parse(grammar, tokens)
        # New signature: parse(tokens, grammar=None)
        # If second arg is a list and first looks like a grammar, old style.
        if tokens_or_none is not None and isinstance(tokens_or_none, list):
            # Old style: parse(grammar, tokens)
            g = tokens_or_grammar
            tokens = tokens_or_none
        else:
            # New style: parse(tokens, grammar=None)
            tokens = tokens_or_grammar
            g = grammar or self._grammar
        if g is None:
            raise ValueError("No grammar provided")
        g_impl = g._impl if hasattr(g, "_impl") else g
        # Reconstruct NLTK-compatible grammar from Rust CFG productions.
        import nltk

        prod_strs = []
        for lhs, rhs in g_impl.productions():
            rhs_parts = []
            for r in rhs:
                if r and r[0].islower():
                    rhs_parts.append(f"'{r}'")
                else:
                    rhs_parts.append(r)
            prod_strs.append(f"{lhs} -> {' '.join(rhs_parts)}")
        nltk_g = nltk.CFG.fromstring("\n".join(prod_strs))
        results = list(nltk.EarleyChartParser(nltk_g).parse(tokens))
        if not results:
            raise ValueError("no parse found")
        return results

    def parse_sents(self, sentences_or_grammar, sentences_or_none=None, grammar=None):
        # Backward compat: old signature was parse_sents(grammar, sentences)
        if sentences_or_none is not None and isinstance(sentences_or_none, list):
            # Old style: parse_sents(grammar, sentences)
            g = sentences_or_grammar
            sentences = sentences_or_none
        else:
            sentences = sentences_or_grammar
            g = grammar or self._grammar
        if g is None:
            raise ValueError("No grammar provided")
        return [self.parse(s, grammar=g) for s in sentences]
