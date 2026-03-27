param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [ValidateSet("cpu", "gpu", "both")]
    [string]$Device = "gpu",

    [ValidateSet("all", "resnet", "sentiment", "uit")]
    [string]$Tasks = "all",

    [ValidateSet("JavaTorch", "your_framework", "pytorch", "both")]
    [string]$Framework = "JavaTorch",

    [string]$OutputDir = "benchmark/results",
    [long]$Seed = 42,

    [switch]$NoDaemon
)

$ErrorActionPreference = "Stop"

function Get-RunConfig {
    param([string]$Task)

    $isResnet = $Task -like "resnet*"

    if ($Mode -eq "quick") {
        if ($isResnet) {
            return @{ epochs = 2; batch = 64; inferWarmup = 5; inferSteps = 10 }
        }
        return @{ epochs = 3; batch = 16; inferWarmup = 5; inferSteps = 20 }
    }

    if ($isResnet) {
        return @{ epochs = 10; batch = 64; inferWarmup = 10; inferSteps = 50 }
    }
    return @{ epochs = 12; batch = 16; inferWarmup = 10; inferSteps = 100 }
}

function Invoke-Benchmark {
    param(
        [string]$GradleTask,
        [string]$TaskName,
        [string]$DeviceName
    )

    $cfg = Get-RunConfig -Task $TaskName
    $runId = "${TaskName}_${DeviceName}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

    $gradleCmd = @(
        ":examples:$GradleTask",
        "--args=--device=$DeviceName --epochs=$($cfg.epochs) --batchSize=$($cfg.batch) --inferWarmup=$($cfg.inferWarmup) --inferSteps=$($cfg.inferSteps) --seed=$Seed --outputDir=$OutputDir --runId=$runId",
        "--console=plain"
    )

    if ($NoDaemon) {
        $gradleCmd += "--no-daemon"
    }

    Write-Host ""
    Write-Host "=== Running $TaskName on $DeviceName (mode=$Mode) ===" -ForegroundColor Cyan
    Write-Host ".\\gradlew.bat $($gradleCmd -join ' ')" -ForegroundColor DarkGray
    Write-Host "Tip: plain console mode is enabled, so epoch progress lines stay visible." -ForegroundColor DarkGray

    & .\gradlew.bat @gradleCmd
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed: $TaskName on $DeviceName"
    }
}

function Invoke-UitMatrix {
    param(
        [string]$FrameworkName,
        [string]$DeviceName
    )

    $seeds = "$Seed"

    Write-Host ""
    Write-Host "=== Running UIT-VSFC ($FrameworkName) on $DeviceName (mode=$Mode) ===" -ForegroundColor Cyan

    if ($FrameworkName -eq "JavaTorch" -or $FrameworkName -eq "your_framework") {
        $scriptPath = "scripts/run-uit-vsfc-matrix.ps1"
        if (!(Test-Path $scriptPath)) {
            throw "Missing script: $scriptPath"
        }
        Write-Host "powershell -ExecutionPolicy Bypass -File $scriptPath -Mode $Mode -Device $DeviceName -Models all -Seeds `"$seeds`" -OutputDir $OutputDir" -ForegroundColor DarkGray
        if ($NoDaemon) {
            powershell -ExecutionPolicy Bypass -File $scriptPath -Mode $Mode -Device $DeviceName -Models all -Seeds $seeds -OutputDir $OutputDir -NoDaemon
        } else {
            powershell -ExecutionPolicy Bypass -File $scriptPath -Mode $Mode -Device $DeviceName -Models all -Seeds $seeds -OutputDir $OutputDir
        }
    } elseif ($FrameworkName -eq "pytorch") {
        $scriptPath = "scripts/run-uit-vsfc-matrix-pytorch.ps1"
        if (!(Test-Path $scriptPath)) {
            throw "Missing script: $scriptPath"
        }
        Write-Host "powershell -ExecutionPolicy Bypass -File $scriptPath -Mode $Mode -Device $DeviceName -Models all -Seeds `"$seeds`" -OutputDir $OutputDir" -ForegroundColor DarkGray
        powershell -ExecutionPolicy Bypass -File $scriptPath -Mode $Mode -Device $DeviceName -Models all -Seeds $seeds -OutputDir $OutputDir
    } else {
        throw "Unsupported framework: $FrameworkName"
    }

    if ($LASTEXITCODE -ne 0) {
        throw "UIT matrix failed for framework=$FrameworkName device=$DeviceName"
    }
}

$devices = @()
if ($Device -eq "both") {
    $devices = @("cpu", "gpu")
} else {
    $devices = @($Device)
}

Write-Host "Benchmark mode: $Mode"
Write-Host "Device set: $($devices -join ', ')"
Write-Host "Task set: $Tasks"
Write-Host "Framework set: $Framework"
Write-Host "Output dir: $OutputDir"
Write-Host "Seed: $Seed"
Write-Host "Gradle daemon: $(if ($NoDaemon) { 'disabled' } else { 'enabled' })"

foreach ($d in $devices) {
    $isJavaFramework = ($Framework -eq "JavaTorch" -or $Framework -eq "your_framework" -or $Framework -eq "both")
    if (($Tasks -eq "all" -or $Tasks -eq "resnet") -and $isJavaFramework) {
        Invoke-Benchmark -GradleTask "benchmarkResNet" -TaskName "resnet_cifar10" -DeviceName $d
    }
    if (($Tasks -eq "all" -or $Tasks -eq "sentiment") -and $isJavaFramework) {
        Invoke-Benchmark -GradleTask "benchmarkSentiment" -TaskName "sentiment_rtpolarity" -DeviceName $d
    }
    if ($Tasks -eq "all" -or $Tasks -eq "uit") {
        if ($isJavaFramework) {
            Invoke-UitMatrix -FrameworkName "JavaTorch" -DeviceName $d
        }
        if ($Framework -eq "pytorch" -or $Framework -eq "both") {
            Invoke-UitMatrix -FrameworkName "pytorch" -DeviceName $d
        }
    }
}

Write-Host ""
Write-Host "Benchmark matrix finished. Artifacts are in: $OutputDir" -ForegroundColor Green
