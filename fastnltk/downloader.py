"""
fastnltk.downloader — Drop-in replacement for nltk.downloader.

Downloads NLTK data files (corpora, tokenizers, taggers).
"""

import nltk.downloader as _nltk_downloader


def download(
    info_or_id: str | None = None,
    download_dir: str | None = None,
    quiet: bool = False,
    force: bool = False,
) -> bool:
    """Download NLTK data.

    Same interface as nltk.download().
    """
    return _nltk_downloader.download(info_or_id, download_dir, quiet, force)


def download_shell():
    """Run the downloader in interactive shell mode."""
    _nltk_downloader.download_shell()


def download_gui():
    """Run the downloader in GUI mode."""
    _nltk_downloader.download_gui()


def update(package: str) -> bool:
    """Update a package."""
    _nltk_downloader.update(package)
