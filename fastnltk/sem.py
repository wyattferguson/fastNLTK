"""
fastnltk.sem — Drop-in replacement for nltk.sem.

Rust-accelerated logic module:
  - Expression parsing (fromstring)
  - Substitution
  - Beta-reduction / simplification
  - Free variable extraction

Other submodules (DRT, boxer, glue, etc.) fall back to NLTK.
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

# Submodule pass-through (complex algorithms, keep as NLTK)
boxer = _nltk_sem.boxer
drt = _nltk_sem.drt
glue = _nltk_sem.glue
hole = _nltk_sem.hole
linearlogic = _nltk_sem.linearlogic
lfg = _nltk_sem.lfg
relextract = _nltk_sem.relextract
chat80 = _nltk_sem.chat80
logic = _nltk_sem.logic
evaluate = _nltk_sem.evaluate
skolemize = _nltk_sem.skolemize
util = _nltk_sem.util

# Re-export logic module's public API
Expression = _nltk_sem.logic.Expression
ApplicationExpression = _nltk_sem.logic.ApplicationExpression
Variable = _nltk_sem.logic.Variable
Constant = _nltk_sem.logic.Constant
LogicalExpressionException = _nltk_sem.logical.LogicalExpressionException
Model = _nltk_sem.evaluate.Model
Valuation = _nltk_sem.evaluate.Valuation
Assignment = _nltk_sem.evaluate.Assignment

__all__ = [
    "fromstring",
    "simplify",
    "Expression",
    "ApplicationExpression",
    "Variable",
    "Constant",
    "Model",
    "Valuation",
    "Assignment",
]


def fromstring(formula, type_check=False):
    """Parse a logical formula string (Rust-accelerated)."""
    if _rust_available:
        return _rust_fromstring(formula)
    return _nltk_sem.logic.Expression.fromstring(formula)


def simplify(formula):
    """Beta-reduce a formula (Rust-accelerated)."""
    if _rust_available:
        return _rust_simplify(formula)
    return _nltk_sem.logic.simplify(formula)
