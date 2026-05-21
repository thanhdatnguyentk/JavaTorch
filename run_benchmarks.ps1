<#
.SYNOPSIS
    JavaTorch Benchmark Suite Runner
    Runs all 5 benchmark tasks: DL4J ResNet, DL4J Sentiment, MemoryPool, ResNet, Sentiment
    Clears old results before each fresh run.

.PARAMETER Device
    Target device: "cpu" or "gpu" (default: "cpu")

.PARAMETER SkipGpu
    Skip GPU-only benchmarks (BenchmarkMemoryPool)

.PARAMETER CleanOnly
    Only clean old results, do not run benchmarks

.EXAMPLE
    .\run_benchmarks.ps1
    .\run_benchmarks.ps1 -Device gpu
    .\run_benchmarks.ps1 -CleanOnly
#>
param(
    [string]$Device = "gpu",
    [switch]$SkipGpu,
    [switch]$CleanOnly
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       JavaTorch Benchmark Suite - $timestamp" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Device    : $Device"
Write-Host "  SkipGpu   : $SkipGpu"
Write-Host "  CleanOnly : $CleanOnly"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ─── Step 1: Clear Old Benchmark Results ────────────────────────────
Write-Host "[1/3] Cleaning old benchmark results..." -ForegroundColor Yellow

$resultsDirs = @(
    "benchmark\results\dl4j",
    "benchmark\results\JavaTorch",
    "benchmark\results\compare",
    "benchmark\results\pytorch"
)

foreach ($dir in $resultsDirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (Test-Path $fullPath) {
        Remove-Item -Recurse -Force $fullPath
        Write-Host "  [DELETED] $dir" -ForegroundColor Red
    } else {
        Write-Host "  [SKIP]    $dir (not found)" -ForegroundColor DarkGray
    }
}

# Also clean any top-level benchmark CSV from SotaBenchmarkRunner
$oldCsv = Join-Path $ProjectRoot "benchmark\benchmark_results.csv"
if (Test-Path $oldCsv) {
    Remove-Item -Force $oldCsv
    Write-Host "  [DELETED] benchmark\benchmark_results.csv" -ForegroundColor Red
}

Write-Host "  [DONE] Old results cleared." -ForegroundColor Green
Write-Host ""

if ($CleanOnly) {
    Write-Host "CleanOnly mode - exiting without running benchmarks." -ForegroundColor Yellow
    exit 0
}

# ─── Step 2: Ensure logs directory ──────────────────────────────────
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$summaryCsv = "benchmark_comparison_summary.csv"
if (-not (Test-Path $summaryCsv)) {
    "Timestamp,Framework,Model,Device,E2E_Time_Seconds,Peak_RAM_MB,Peak_VRAM_MB" | Out-File $summaryCsv -Encoding UTF8
}

# ─── Step 3: Define Benchmark Tasks ────────────────────────────────
# Order: light NLP -> heavy CV -> GPU-only stress
$benchmarks = @(
    @{
        Name      = "DL4J Sentiment (RT-Polarity)"
        Task      = "benchmarkDl4jSentiment"
        GpuOnly   = $false
        BenchArgs = "--device cpu"
    },
    @{
        Name      = "JavaTorch Sentiment (RT-Polarity)"
        Task      = "benchmarkSentiment"
        GpuOnly   = $false
        BenchArgs = "--device $Device"
    },
    @{
        Name      = "DL4J ResNet (CIFAR-10)"
        Task      = "benchmarkDl4jResNet"
        GpuOnly   = $false
        BenchArgs = "--device cpu"
    },
    @{
        Name      = "JavaTorch ResNet-18 (CIFAR-10)"
        Task      = "benchmarkResNet"
        GpuOnly   = $false
        BenchArgs = "--device $Device"
    },
    @{
        Name      = "GPU MemoryPool Stress Test"
        Task      = "benchmarkMemoryPool"
        GpuOnly   = $true
        BenchArgs = ""
    }
)

# ─── Step 4: Run Benchmarks ─────────────────────────────────────────
$totalStart = Get-Date
$results = @()
$index = 0

foreach ($bm in $benchmarks) {
    $index++

    # Skip GPU-only benchmarks if requested
    if ($bm.GpuOnly -and ($SkipGpu -or $Device -eq "cpu")) {
        Write-Host "[$index/5] SKIPPED: $($bm.Name) (GPU-only, current device=$Device)" -ForegroundColor DarkYellow
        $results += @{ Name = $bm.Name; Status = "SKIPPED"; TimeMs = 0 }
        continue
    }

    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor White
    Write-Host "[$index/5] Starting: $($bm.Name)" -ForegroundColor Cyan
    Write-Host "     Gradle task: :examples:$($bm.Task)" -ForegroundColor DarkGray
    Write-Host "     Args: $($bm.BenchArgs)" -ForegroundColor DarkGray
    Write-Host "------------------------------------------------------------" -ForegroundColor White

    $logFile = "logs\benchmark_$($bm.Task)_$timestamp.log"
    $bmStart = Get-Date

    try {
        $gradleCmd = ".\gradlew.bat :examples:$($bm.Task) --no-daemon"
        if ($bm.BenchArgs -ne "") {
            $gradleCmd += " `"-PbenchArgs=$($bm.BenchArgs)`""
        }

        # Start Monitor
        $stopFile = "$PWD\stop_monitor.tmp"
        if (Test-Path $stopFile) { Remove-Item $stopFile }
        $job = Start-Job -ScriptBlock {
            param($stopFile)
            $maxRam = 0
            $maxVram = 0
            while (-not (Test-Path $stopFile)) {
                try {
                    $procs = Get-Process -Name "java" -ErrorAction SilentlyContinue
                    if ($procs) {
                        $ram = ($procs | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB
                        if ($ram -gt $maxRam) { $maxRam = $ram }
                    }
                } catch {}
                try {
                    $vramOutput = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
                    if ($vramOutput) {
                        $vram = [int]($vramOutput[0].Trim())
                        if ($vram -gt $maxVram) { $maxVram = $vram }
                    }
                } catch {}
                Start-Sleep -Milliseconds 500
            }
            return @{ MaxRam = $maxRam; MaxVram = $maxVram }
        } -ArgumentList $stopFile

        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        # Execute via Invoke-Expression to handle the -P argument correctly
        Invoke-Expression "$gradleCmd 2>&1" | Tee-Object -FilePath $logFile

        $sw.Stop()
        Out-File -FilePath $stopFile -InputObject "stop"
        Wait-Job $job | Out-Null
        $jobResult = Receive-Job $job
        Remove-Job $job
        if (Test-Path $stopFile) { Remove-Item $stopFile }
    
        $peakRam = [math]::Round($jobResult.MaxRam, 2)
        $peakVram = $jobResult.MaxVram
        $e2eTime = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "$ts,JavaTorch,$($bm.Name),$Device,$e2eTime,$peakRam,$peakVram" | Out-File $summaryCsv -Encoding UTF8 -Append

        $exitCode = $LASTEXITCODE
        $bmEnd = Get-Date
        $bmDuration = ($bmEnd - $bmStart).TotalMilliseconds

        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Host "  >> $($bm.Name) PASSED in $([math]::Round($bmDuration / 1000, 1))s" -ForegroundColor Green
            $results += @{ Name = $bm.Name; Status = "PASSED"; TimeMs = [int]$bmDuration }
        } else {
            Write-Host ""
            Write-Host "  >> $($bm.Name) FAILED (exit code $exitCode)" -ForegroundColor Red
            $results += @{ Name = $bm.Name; Status = "FAILED"; TimeMs = [int]$bmDuration }
        }
    } catch {
        $bmEnd = Get-Date
        $bmDuration = ($bmEnd - $bmStart).TotalMilliseconds
        Write-Host ""
        Write-Host "  >> $($bm.Name) EXCEPTION: $_" -ForegroundColor Red
        $results += @{ Name = $bm.Name; Status = "ERROR"; TimeMs = [int]$bmDuration }
    }

    Write-Host "     Log saved to: $logFile" -ForegroundColor DarkGray

    # GC and cooldown between runs
    [System.GC]::Collect()
    Start-Sleep -Seconds 3
}

# ─── Step 5: Summary Report ─────────────────────────────────────────
$totalEnd = Get-Date
$totalDuration = ($totalEnd - $totalStart).TotalSeconds

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       BENCHMARK SUITE RESULTS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0
$skipped = 0

foreach ($r in $results) {
    $statusColor = switch ($r.Status) {
        "PASSED"  { "Green" }
        "FAILED"  { "Red" }
        "ERROR"   { "Red" }
        "SKIPPED" { "Yellow" }
        default   { "White" }
    }
    $timeStr = if ($r.TimeMs -gt 0) { "$([math]::Round($r.TimeMs / 1000, 1))s" } else { "-" }
    Write-Host ("  [{0,-7}] {1,-40} {2}" -f $r.Status, $r.Name, $timeStr) -ForegroundColor $statusColor

    switch ($r.Status) {
        "PASSED"  { $passed++ }
        "FAILED"  { $failed++ }
        "ERROR"   { $failed++ }
        "SKIPPED" { $skipped++ }
    }
}

Write-Host ""
Write-Host "  Total: $($results.Count) | Passed: $passed | Failed: $failed | Skipped: $skipped" -ForegroundColor White
Write-Host "  Total time: $([math]::Round($totalDuration, 1))s" -ForegroundColor White
Write-Host ""
Write-Host "  Results saved to: benchmark\results\" -ForegroundColor DarkGray
Write-Host "  Logs saved to:    logs\" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

# Exit with error code if any benchmark failed
if ($failed -gt 0) {
    exit 1
}
