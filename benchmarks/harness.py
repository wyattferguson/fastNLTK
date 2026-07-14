"""
Benchmark harness — automatic regression benchmarking for fastNLTK.

Usage:
    python -m benchmarks.run                          # Run all, print table
    python -m benchmarks.run --save results/latest.json   # Run + save
    python -m benchmarks.run --regression results/latest.json  # Compare against baseline
    python -m benchmarks.run --regression results/latest.json --threshold 0.20  # 20% slack

Exit code: 0 if all pass, 1 if any regression exceeds threshold.
"""

import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT)

# ── Data fixtures ─────────────────────────────────────────

_FIXTURES: dict[str, str] = {}


def _load_fixtures():
    """Load text data files on first access."""
    if _FIXTURES:
        return
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_DIR, fname)
            key = os.path.splitext(fname)[0]
            with open(path, encoding="utf-8") as f:
                _FIXTURES[key] = f.read()


def fixture(key: str) -> str:
    _load_fixtures()
    if key not in _FIXTURES:
        raise FileNotFoundError(f"No fixture '{key}' in {DATA_DIR}/")
    return _FIXTURES[key]


# ── Benchmark result types ───────────────────────────────


@dataclass
class BenchResult:
    """Single benchmark measurement."""

    name: str
    group: str
    params: dict[str, Any] = field(default_factory=dict)
    nltk_ms: float = 0.0
    fast_ms: float = 0.0
    speedup: float = 0.0
    fast_only_ms: float = 0.0  # for fastNLTK-only benchmarks (no NLTK equivalent)
    iterations: int = 0
    version: str = ""  # git describe or commit hash


@dataclass
class BenchSuite:
    """Collection of benchmark results."""

    timestamp: str = ""
    git_hash: str = ""
    results: list[BenchResult] = field(default_factory=list)


# ── Timer helpers ─────────────────────────────────────────


def _median_time(fn: Callable, iterations: int = 30, warmup: int = 3) -> float:
    """Run fn `iterations` times, return median wall time in ms."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _git_hash() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", PROJECT, "describe", "--always", "--dirty"],
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


# ── Regression check ──────────────────────────────────────


REGRESSION_THRESHOLD = 0.25  # 25% — accounts for system noise in microbenchmarks
# Absolute noise floor: skip regression checks for benchmarks below this threshold.
# Sub-millisecond microbenchmarks are dominated by timer resolution / OS jitter
# and cannot reliably stay within the 25% relative threshold.
MIN_ABSOLUTE_REGRESSION_MS = 0.5


def check_regression(
    current: list[BenchResult],
    baseline: list[BenchResult],
    threshold: float = REGRESSION_THRESHOLD,
) -> list[tuple[BenchResult, BenchResult, float]]:
    """Compare current results vs baseline. Return list of regressions.

    Each tuple: (current, baseline, fractional_change)
    Positive change = slower (regression).
    """
    base_map: dict[str, BenchResult] = {}
    for r in baseline:
        key = f"{r.group}:{r.name}"
        base_map[key] = r

    regressions: list[tuple[BenchResult, BenchResult, float]] = []
    for cur in current:
        key = f"{cur.group}:{cur.name}"
        base = base_map.get(key)
        if base is None:
            continue  # new benchmark, no baseline
        # Use fast_only_ms if available (no NLTK comparison), else fast_ms
        cur_time = cur.fast_only_ms or cur.fast_ms
        base_time = base.fast_only_ms or base.fast_ms
        if base_time == 0:
            continue
        # Skip microbenchmarks where noise dominates the signal
        if cur_time < MIN_ABSOLUTE_REGRESSION_MS and base_time < MIN_ABSOLUTE_REGRESSION_MS:
            continue
        change = (cur_time - base_time) / base_time
        if change > threshold:
            regressions.append((cur, base, change))
    return regressions


# ── JSON I/O ──────────────────────────────────────────────


def save_results(results: list[BenchResult], path: str = "") -> str:
    """Save results to JSON. Returns path written."""
    # Detect whether release or debug build
    import subprocess

    build_type = "release"
    try:
        r = subprocess.run(
            ["cargo", "metadata", "--format-version", "1"],
            capture_output=True,
            text=True,
            cwd=PROJECT,
            timeout=5,
        )
        if r.returncode == 0:
            meta = json.loads(r.stdout)
            target_dir = meta.get("target_directory", "")
            profile = "release" if "/release/" in target_dir.replace("\\", "/") else "debug"
            if profile:
                build_type = profile
    except Exception:
        pass

    suite = BenchSuite(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_hash=f"{_git_hash()}+{build_type}",
        results=results,
    )
    if not path:
        path = os.path.join(RESULTS_DIR, f"bench_{suite.git_hash}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(suite), f, indent=2, default=str)
    return path


def load_results(path: str) -> BenchSuite:
    with open(path) as f:
        data = json.load(f)
    return BenchSuite(
        timestamp=data.get("timestamp", ""),
        git_hash=data.get("git_hash", ""),
        results=[BenchResult(**r) for r in data.get("results", [])],
    )


# ── Table printer ─────────────────────────────────────────


def print_table(results: list[BenchResult], title: str = "Benchmarks"):
    """Print aligned results table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Benchmark':<45} {'NLTK(ms)':>10} {'fast(ms)':>10} {'Speedup':>8}  {'Hit':>5}")
    print(f"  {'-' * 45} {'-' * 10} {'-' * 10} {'-' * 8}  {'-' * 5}")
    for r in results:
        nltk_s = f"{r.nltk_ms:.2f}" if r.nltk_ms else "-"
        fast_s = f"{r.fast_only_ms:.4f}" if r.fast_only_ms else f"{r.fast_ms:.2f}"
        if r.fast_only_ms:
            # fastNLTK only benchmark
            print(f"  {r.name:<45} {'-':>10} {fast_s:>10} {'-':>8}  {'-':>5}")
        else:
            sp = r.speedup
            hit = "OK" if sp >= 1.5 else "--"
            print(f"  {r.name:<45} {nltk_s:>10} {fast_s:>10} {sp:>7.1f}x  {hit:>5}")
    print(f"{'=' * 80}")


def print_regression_table(regressions: list[tuple[BenchResult, BenchResult, float]]):
    """Print regression details."""
    if not regressions:
        print("\n  [OK] No regressions detected.\n")
        return
    print(f"\n  {'!' * 60}")
    print(f"  REGRESSIONS DETECTED ({len(regressions)})")
    print(f"  {'!' * 60}")
    print(f"  {'Benchmark':<45} {'Before(ms)':>10} {'After(ms)':>10} {'Change':>8}")
    print(f"  {'-' * 45} {'-' * 10} {'-' * 10} {'-' * 8}")
    for cur, base, change in regressions:
        cur_t = cur.fast_only_ms or cur.fast_ms
        base_t = base.fast_only_ms or base.fast_ms
        print(
            f"  {cur.name:<45} {base_t:>10.4f} {cur_t:>10.4f} {'+' if change > 0 else ''}{change * 100:>7.1f}%"
        )
    print()
