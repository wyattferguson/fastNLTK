# Quality checks — same as CI workflow
# Run: .\check.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 1/6  cargo fmt --check ===" -ForegroundColor Cyan
cargo fmt --all -- --check
Write-Host "  PASS" -ForegroundColor Green

Write-Host "=== 2/6  cargo clippy ===" -ForegroundColor Cyan
$env:PYO3_PYTHON = "python"
cargo clippy --all-targets --no-default-features --features parallel -- -D clippy::correctness -D clippy::suspicious -D clippy::perf -W clippy::style -W clippy::complexity
Write-Host "  PASS" -ForegroundColor Green

Write-Host "=== 3/6  cargo doc ===" -ForegroundColor Cyan
$env:RUSTDOCFLAGS = "-D warnings"
cargo doc --no-deps --document-private-items
Write-Host "  PASS" -ForegroundColor Green

Write-Host "=== 4/6  cargo test ===" -ForegroundColor Cyan
$env:PYO3_PYTHON = "python"
cargo test --no-default-features --features parallel
Write-Host "  PASS" -ForegroundColor Green

Write-Host "=== 5/6  cargo deny check ===" -ForegroundColor Cyan
cargo deny check --all-features
Write-Host "  PASS" -ForegroundColor Green

Write-Host "=== 6/6  Python tests ===" -ForegroundColor Cyan
python -m pytest tests/ -q --tb=short -k "not test_coverage"
python -m pytest tests/test_invariants.py -q
Write-Host "  PASS" -ForegroundColor Green

Write-Host ""
Write-Host "All checks passed!" -ForegroundColor Green
