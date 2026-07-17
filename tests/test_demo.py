"""Smoke test: run demo.py end-to-end as a full-module integration check."""

import importlib.util
import io
import sys
import types


def test_demo_smoke():
    """Run demo.main() and verify all 21 modules execute without error."""
    # Load demo.py as a module so it uses the in-tree fastnltk
    spec = importlib.util.spec_from_file_location("demo", "demo.py")
    mod = importlib.util.module_from_spec(spec)

    # Capture stdout to avoid polluting test output
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()

    # Every module should have a benchmark entry
    sections = [
        "tokenize",
        "tag",
        "chunk",
        "stem",
        "probability",
        "collocations",
        "sentiment",
        "tree",
        "metrics",
        "lm",
        "classify",
        "cluster",
        "parse",
        "ccg",
        "chat",
        "inference",
        "sem",
        "translate",
        "ngrams",
        "downloader",
        "corpus",
    ]
    for name in sections:
        assert name in output, f"Module '{name}' missing from demo output"

    # Benchmark summary present
    assert "BENCHMARK SUMMARY" in output
    assert "TOTAL" in output
    assert "MODULES" in output

    # All modules accounted for
    assert "MODULES" in output
