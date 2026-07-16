"""Pytest configuration for fastNLTK tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-nltk-compat",
        action="store_true",
        default=False,
        help="Run NLTK compatibility tests (requires nltk data)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nltk_compat: mark test as requiring NLTK compatibility comparison",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-nltk-compat"):
        skip_nltk = pytest.mark.skip(reason="use --run-nltk-compat to run NLTK compatibility tests")
        for item in items:
            if "nltk_compat" in item.keywords:
                item.add_marker(skip_nltk)
