"""Add type annotations to remaining Python shim functions."""
import re
import os

files = [
    "fastnltk/tag.py",
    "fastnltk/chunk.py",
    "fastnltk/classify.py",
    "fastnltk/collocations.py",
    "fastnltk/cluster.py",
    "fastnltk/chat.py",
    "fastnltk/lm.py",
    "fastnltk/metrics.py",
    "fastnltk/parse.py",
    "fastnltk/probability.py",
    "fastnltk/sem.py",
    "fastnltk/sentiment.py",
    "fastnltk/translate.py",
    "fastnltk/tree.py",
]

for path in files:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()

    changed = False

    # Tag methods
    c, n = re.subn(
        r"^    def tag\(self, tokens\):",
        r"    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def tag\(self, words\):",
        r"    def tag(self, words: list[str]) -> list[tuple[str, str]]:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def tag_sents\(self, sentences\):",
        r"    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def train\(self, sentences\):",
        r"    def train(self, sentences: list[list[str]]) -> None:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def evaluate\(self, gold\):",
        r"    def evaluate(self, gold: list[list[str]]) -> float:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    # Classify methods
    c, n = re.subn(
        r"^    def classify\(self, features\):",
        r"    def classify(self, features: dict) -> str:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def labels\(self\):",
        r"    def labels(self) -> list[str]:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^def pos_tag\(tokens,",
        r"def pos_tag(tokens: list[str],",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^def pos_tag_sents\(sentences,",
        r"def pos_tag_sents(sentences: list[list[str]],",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    # tokenize/span_tokenize (for files that have them)
    c, n = re.subn(
        r"^    def tokenize\(self, text\):",
        r"    def tokenize(self, text: str) -> list[str]:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def span_tokenize\(self, text\):",
        r"    def span_tokenize(self, text: str) -> list[tuple[int, int]]:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def detokenize\(self, tokens\):",
        r"    def detokenize(self, tokens: list[str]) -> str:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    c, n = re.subn(
        r"^    def stem\(self, word\):",
        r"    def stem(self, word: str) -> str:",
        c,
        flags=re.MULTILINE,
    )
    changed = changed or n > 0

    if changed:
        if "from __future__ import annotations" not in c:
            c = "from __future__ import annotations\n\n" + c
        with open(path, "w") as f:
            f.write(c)
        print(f"annotated: {path}")
