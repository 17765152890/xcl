# Build everything (the e2e test needs the test title + DLL) and run all tests.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
# tests on game data pass without running anything when their variable is not set
$missing = 'XCL_SAMPLE_RPF', 'XCL_SAMPLE_DLLS', 'XCL_SAMPLE_TTPF', 'XCL_SAMPLE_BK2', 'XCL_SAMPLE_BVH4', 'XCL_SAMPLE_AWC', 'XCL_SAMPLE_XG', 'XCL_SAMPLE_DXC', 'XCL_SAMPLE_FFX' | Where-Object { -not (Test-Path "env:$_") }
if ($missing) { Write-Host "not set, their tests are skipped: $($missing -join ', ')" }
cargo build --workspace
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo test --workspace -- --test-threads=4
exit $LASTEXITCODE
