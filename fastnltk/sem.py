"""
fastnltk.sem — Drop-in replacement for nltk.sem.

Rust-accelerated logic module: expression parsing, simplification,
model evaluation. DRT (Discourse Representation Theory).
Submodules (boxer, glue, etc.) fall back to NLTK.
"""

import warnings

import nltk.sem as _nltk_sem

_rust_available = False
try:
    from fastnltk._rust import fromstring as _rust_fromstring
    from fastnltk._rust import simplify as _rust_simplify
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to NLTK sem"
    )

# Submodule pass-through (version-safe)
for _attr in ["boxer", "drt", "glue", "linearlogic", "lfg", "relextract",
              "chat80", "logic", "evaluate", "skolemize", "util", "hole"]:
    if hasattr(_nltk_sem, _attr):
        globals()[_attr] = getattr(_nltk_sem, _attr)

# Core types from logic submodule
Expression = _nltk_sem.logic.Expression
ApplicationExpression = _nltk_sem.logic.ApplicationExpression
Variable = _nltk_sem.logic.Variable
LogicalExpressionException = _nltk_sem.logic.LogicalExpressionException

__all__ = [
    "fromstring",
    "simplify",
    "Expression",
    "ApplicationExpression",
    "Variable",
    "LogicalExpressionException",
]


def fromstring(formula, type_check=False):
    if _rust_available:
        return _rust_fromstring(formula)
    return _nltk_sem.logic.Expression.fromstring(formula)


def simplify(formula):
    if _rust_available:
        return _rust_simplify(formula)
    return _nltk_sem.logic.simplify(formula)
