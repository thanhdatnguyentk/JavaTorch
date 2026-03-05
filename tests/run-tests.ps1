# Auto test runner for ML_framework
# Usage: PowerShell -ExecutionPolicy Bypass -File tests\run-tests.ps1

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $root

Write-Host "Project root: $($pwd)"

# Ensure bin exists
if (!(Test-Path bin)) { New-Item -ItemType Directory -Path bin | Out-Null }

Write-Host "Compiling library sources..."
javac --add-modules jdk.incubator.vector -d bin src\com\user\nn\*.java src\*.java
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation of sources failed."; exit 1 }

Write-Host "Compiling test runners..."
javac --add-modules jdk.incubator.vector -d bin -cp bin tests\java\com\user\nn\*.java
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation of tests failed."; exit 1 }

$tests = @(
    'com.user.nn.TestMatOps',
    'com.user.nn.TestContainers',
    'com.user.nn.TestParameterAndModules',
    'com.user.nn.TestFunctional',
    'com.user.nn.TestLinearReLU',
    'com.user.nn.TestActivations',
    'com.user.nn.TestLossesAndNorms',
    'com.user.nn.TestConvPool',
    'com.user.nn.TestTorchCoverage',
    'com.user.nn.TestTorchExtras',
    'com.user.nn.TestTensor',
    'com.user.nn.TestGatherScatterExtras'
    , 'com.user.nn.TestAutogradSimple'
    , 'com.user.nn.TestAutogradLinear'
    , 'com.user.nn.TestAutogradShapeOps'
    , 'com.user.nn.TestAutogradReductions'
    , 'com.user.nn.TestAutogradMatmul'
    , 'com.user.nn.TestAutogradActivations'
    , 'com.user.nn.TestAutogradMLP'
    , 'com.user.nn.TestAutogradConv'
    , 'com.user.nn.TestOptimizers'
    , 'com.user.nn.TestLossFunctions'
    , 'com.user.nn.TestNormLayers'
)

$failures = @()

foreach ($t in $tests) {
    Write-Host "\n=== Running $t ==="
    java --add-modules jdk.incubator.vector -cp bin $t
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host "PASS: $t" -ForegroundColor Green
    } else {
        Write-Host "FAIL: $t (exit $code)" -ForegroundColor Red
        $failures += "$t (exit $code)"
    }
}

Write-Host "\nTest summary:"
if ($failures.Count -eq 0) {
    Write-Host "All tests passed." -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some tests failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host " - $_" }
    exit 2
}
