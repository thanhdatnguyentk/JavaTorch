param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [ValidateSet("cpu", "gpu", "both")]
    [string]$Device = "cpu",

    [ValidateSet("JavaTorch", "your_framework", "pytorch", "both")]
    [string]$Framework = "both",

    [ValidateSet("all", "lstm", "transformer")]
    [string]$Models = "all",

    [string]$Seeds = "42",
    [string]$ResultsDir = "benchmark/results",
    [string]$OutputDir = "benchmark/results/compare/uit_vsfc_multitask",

    [double]$AccuracyWeight = 0.5,
    [double]$ThroughputWeight = 0.3,
    [double]$LatencyWeight = 0.2,

    [bool]$LatestOnly = $true,
    [bool]$DropMissingMetrics = $true,
    [bool]$RejectInvalid = $true,

    [switch]$RunBenchmarks,
    [switch]$NoDaemon,
    [string]$SeedFilter = ""
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Get-PythonExe {
    $candidate = ".venv/Scripts/python.exe"
    if (Test-Path $candidate) {
        return $candidate
    }
    return "python"
}

function Invoke-UitBenchmarks {
    if (-not $RunBenchmarks) {
        return
    }

    Write-Section "Phase 4 | Benchmark orchestration"

    if ($Models -eq "all") {
        $seedSingle = 42
        try {
            $seedSingle = [long](($Seeds -split ",")[0].Trim())
        } catch {
            Write-Warning "Could not parse first seed from '$Seeds', fallback to 42."
            $seedSingle = 42
        }

        $args = @(
            "-Mode", $Mode,
            "-Device", $Device,
            "-Tasks", "uit",
            "-Framework", $Framework,
            "-OutputDir", $ResultsDir,
            "-Seed", "$seedSingle"
        )
        if ($NoDaemon) {
            $args += "-NoDaemon"
        }

        Write-Host "./scripts/run-benchmark-matrix.ps1 $($args -join ' ')" -ForegroundColor DarkGray
        & ./scripts/run-benchmark-matrix.ps1 @args
        if ($LASTEXITCODE -ne 0) {
            throw "UIT benchmark orchestration failed via run-benchmark-matrix.ps1"
        }
        return
    }

    if ($Framework -eq "JavaTorch" -or $Framework -eq "your_framework" -or $Framework -eq "both") {
        $argsJava = @(
            "-Mode", $Mode,
            "-Device", $Device,
            "-Models", $Models,
            "-Seeds", $Seeds,
            "-OutputDir", $ResultsDir
        )
        if ($NoDaemon) {
            $argsJava += "-NoDaemon"
        }
        Write-Host "./scripts/run-uit-vsfc-matrix.ps1 $($argsJava -join ' ')" -ForegroundColor DarkGray
        & ./scripts/run-uit-vsfc-matrix.ps1 @argsJava
        if ($LASTEXITCODE -ne 0) {
            throw "UIT Java benchmark matrix failed"
        }
    }

    if ($Framework -eq "pytorch" -or $Framework -eq "both") {
        $argsTorch = @(
            "-Mode", $Mode,
            "-Device", $Device,
            "-Models", $Models,
            "-Seeds", $Seeds,
            "-OutputDir", $ResultsDir
        )
        Write-Host "./scripts/run-uit-vsfc-matrix-pytorch.ps1 $($argsTorch -join ' ')" -ForegroundColor DarkGray
        & ./scripts/run-uit-vsfc-matrix-pytorch.ps1 @argsTorch
        if ($LASTEXITCODE -ne 0) {
            throw "UIT PyTorch benchmark matrix failed"
        }
    }
}

function Invoke-AggregationAndVisualization {
    Write-Section "Phase 4 | Aggregate and visualize"

    $py = Get-PythonExe
    $aggArgs = @(
        "scripts/aggregate_uit_vsfc_compare.py",
        "--resultsDir", $ResultsDir,
        "--outputDir", $OutputDir,
        "--accuracyWeight", "$AccuracyWeight",
        "--throughputWeight", "$ThroughputWeight",
        "--latencyWeight", "$LatencyWeight"
    )

    if ($LatestOnly) {
        $aggArgs += "--latestOnly"
    }
    if ($DropMissingMetrics) {
        $aggArgs += "--dropMissingMetrics"
    }
    if ($RejectInvalid) {
        $aggArgs += "--rejectInvalid"
    }

    Write-Host "$py $($aggArgs -join ' ')" -ForegroundColor DarkGray
    & $py @aggArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed"
    }

    $csvPath = Join-Path $OutputDir "comparison.csv"
    if (!(Test-Path $csvPath)) {
        throw "Missing comparison.csv after aggregation: $csvPath"
    }

    $vizArgs = @(
        "scripts/visualize_uit_vsfc_compare.py",
        "--csv", $csvPath,
        "--outDir", $OutputDir,
        "--qualityWeight", "$AccuracyWeight",
        "--latencyWeight", "$LatencyWeight",
        "--throughputWeight", "$ThroughputWeight"
    )

    if ($SeedFilter -ne $null -and $SeedFilter.Trim().Length -gt 0) {
        $vizArgs += @("--seed", $SeedFilter)
    }

    Write-Host "$py $($vizArgs -join ' ')" -ForegroundColor DarkGray
    & $py @vizArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Visualization failed"
    }
}

function Write-Phase4Summary {
    Write-Section "Phase 4 | Write report summary"

    $comparisonCsv = Join-Path $OutputDir "comparison.csv"
    $rankingCsv = Join-Path $OutputDir "normalized_ranking.csv"
    $invalidCsv = Join-Path $OutputDir "invalid_rows.csv"
    $reportPath = Join-Path $OutputDir ("phase4_report_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".md")

    $rows = @()
    if (Test-Path $comparisonCsv) {
        $rows = Import-Csv $comparisonCsv
    }

    $topComp = @()
    if ($rows.Count -gt 0) {
        $topComp = $rows | Sort-Object {[double]$_.score_overall} -Descending | Select-Object -First 5
    }

    $topNorm = @()
    if (Test-Path $rankingCsv) {
        $topNorm = Import-Csv $rankingCsv | Select-Object -First 5
    }

    $invalidCount = 0
    if (Test-Path $invalidCsv) {
        $invalidCount = (Import-Csv $invalidCsv).Count
    }

    $lines = @()
    $lines += "## UIT-VSFC Phase 4 Report"
    $lines += ""
    $lines += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $lines += "Mode: $Mode | Device: $Device | Framework: $Framework | Models: $Models | Seeds: $Seeds"
    $lines += "ResultsDir: $ResultsDir"
    $lines += "OutputDir: $OutputDir"
    $lines += "Scoring weights: accuracy=$AccuracyWeight throughput=$ThroughputWeight latency=$LatencyWeight"
    $lines += "Sanity checks: latestOnly=$LatestOnly dropMissingMetrics=$DropMissingMetrics rejectInvalid=$RejectInvalid invalidRows=$invalidCount"
    $lines += ""
    $lines += "### Top 5 by aggregate score (comparison.csv)"
    if ($topComp.Count -eq 0) {
        $lines += "No rows available."
    } else {
        $lines += "| Rank | Framework | Run ID | Model | Device | Seed | score_overall | weighted_test_f1 | p50_ms | throughput_sps |"
        $lines += "|---|---|---|---|---|---:|---:|---:|---:|---:|"
        $rank = 1
        foreach ($r in $topComp) {
            $lines += "| $rank | $($r.framework) | $($r.run_id) | $($r.model) | $($r.device) | $($r.seed) | $($r.score_overall) | $($r.weighted_test_f1) | $($r.inference_p50_ms) | $($r.inference_throughput_sps) |"
            $rank++
        }
    }

    $lines += ""
    $lines += "### Top 5 by normalized_overall (normalized_ranking.csv)"
    if ($topNorm.Count -eq 0) {
        $lines += "No normalized ranking rows available."
    } else {
        $lines += "| Rank | Framework | Run ID | Model | Device | Seed | normalized_overall | weighted_test_f1 | p50_ms | throughput_sps |"
        $lines += "|---|---|---|---|---|---:|---:|---:|---:|---:|"
        foreach ($r in $topNorm) {
            $lines += "| $($r.rank) | $($r.framework) | $($r.run_id) | $($r.model) | $($r.device) | $($r.seed) | $($r.normalized_overall) | $($r.weighted_test_f1) | $($r.inference_p50_ms) | $($r.inference_throughput_sps) |"
        }
    }

    $lines += ""
    $lines += "Artifacts:"
    $lines += "- comparison.csv"
    $lines += "- comparison.md"
    $lines += "- normalized_ranking.csv"
    $lines += "- weighted_test_f1.png"
    $lines += "- quality_breakdown.png"
    $lines += "- speed_vs_quality.png"
    $lines += "- throughput.png"
    $lines += "- normalized_overall.png"
    if (Test-Path $invalidCsv) {
        $lines += "- invalid_rows.csv"
    }

    Set-Content -Path $reportPath -Value ($lines -join "`n") -Encoding utf8
    Write-Host "Wrote: $reportPath" -ForegroundColor Green
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"
Write-Host "Phase 4 Orchestration"

Invoke-UitBenchmarks
Invoke-AggregationAndVisualization
Write-Phase4Summary

Write-Host "`nPhase 4 completed." -ForegroundColor Green