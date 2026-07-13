#!/usr/bin/env python
"""
Update the benchmark table in README.md with results from --benchmark-json.

Usage:
    python scripts/update_benchmark_table.py benchmarks/results/<file>.json
    python scripts/update_benchmark_table.py benchmarks/results/*.json  # merge all
"""

import json
import re
import sys
from pathlib import Path

README = Path("README.md")
SECTION_MARKERS = {
    "tokenize": "### Tokenization",
    "stem": "### Stemming",
    "tag": "### POS Tagging",
    "classify": "### Classification",
    "collocations": "### Collocations & Probability",
    "lm": "### Language Models",
    "pipeline": "### Full Pipeline",
}


def extract_results(json_files):
    """Parse benchmark JSON files and extract speedup data."""
    results = {}
    for f in json_files:
        with open(f) as fp:
            data = json.load(fp)
        for bench in data.get("benchmarks", []):
            group = bench.get("group", "unknown")
            name = bench.get("name", "")
            params = bench.get("params", {})
            size = params.get("size", "")
            stats = bench.get("stats", {})
            mean = stats.get("mean", 0) * 1000  # seconds → ms

            key = (group, name.split("::")[-1], size)
            if "nltk" in name.lower():
                results.setdefault(key, {})["nltk_ms"] = round(mean, 2)
            else:
                results.setdefault(key, {})["fastnltk_ms"] = round(mean, 2)
                results[key]["date"] = data.get("commit_date", "—")
    return results


def format_speedup(results):
    """Format results into markdown table rows."""
    rows = []
    for (group, func, size), vals in sorted(results.items()):
        nltk_ms = vals.get("nltk_ms", "—")
        fastnltk_ms = vals.get("fastnltk_ms", "—")
        date = vals.get("date", "—")
        if isinstance(nltk_ms, (int, float)) and isinstance(fastnltk_ms, (int, float)):
            speedup = f"{nltk_ms / fastnltk_ms:.1f}x" if fastnltk_ms > 0 else "N/A"
            rows.append((group, (f"| `{func}` | {size} | {nltk_ms} | {fastnltk_ms} | {speedup} | {date} |")))
        else:
            rows.append((group, (f"| `{func}` | {size} | {nltk_ms} | {fastnltk_ms} | — | {date} |")))
    return rows


def main():
    json_files = sys.argv[1:]
    if not json_files:
        print("Usage: update_benchmark_table.py <benchmark.json> [...]")
        sys.exit(1)

    results = extract_results(json_files)
    rows = format_speedup(results)

    if not README.exists():
        print(f"README.md not found: {README}")
        sys.exit(1)

    content = README.read_text()

    for section_title in SECTION_MARKERS.values():
        section_rows = [r for g, r in rows if g == section_title.split()[-1].lower() or any(
            g.lower().startswith(s) for s in ["sent", "word", "regexp", "space", "tab", "line"]
        )]
        if not section_rows:
            continue

        # Find the table under this section and append rows
        # Simple approach: find the first empty line after the section marker
        lines = content.split("\n")
        in_section = False
        for i, line in enumerate(lines):
            if section_title in line:
                in_section = True
                continue
            if in_section and line.strip().startswith("| ---"):
                # Find where table ends
                for j in range(i + 1, len(lines)):
                    if not lines[j].strip().startswith("|"):
                        # Insert new rows before the blank line
                        insert_pos = j
                        break
                else:
                    insert_pos = len(lines)

                # Only add rows not already present
                existing = set(lines[i + 1:insert_pos])
                new_rows = [r for r in section_rows if r not in existing]
                if new_rows:
                    lines[insert_pos:insert_pos] = [r for _, r in new_rows]
                break

    README.write_text("\n".join(lines))
    print(f"Updated {README} with {len(rows)} benchmark rows.")


if __name__ == "__main__":
    main()
