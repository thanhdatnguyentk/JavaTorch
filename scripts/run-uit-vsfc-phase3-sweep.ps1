param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [ValidateSet("cpu", "gpu", "both")]
    [string]$Device = "cpu",

    [ValidateSet("all", "lstm", "transformer")]
    [string]$Models = "transformer",

    [string]$Seeds = "42",
    [string]$DataDir = "examples/data/uit-vsfc",
    [string]$OutputDir = "benchmark/results",

    [string]$LearningRates = "0.001,0.0005",
    [string]$MinLearningRates = "0.0001",
    [ValidateSet("none", "cosine")]
    [string]$LrSchedule = "cosine",
    [string]$WarmupEpochs = "0,2",
    [string]$EarlyStoppingPatience = "0,3",
    [float]$EarlyStoppingMinDelta = 0.0005,

    [string]$BatchSizes = "256,384",
    [string]$MaxLens = "48,64",
    [int]$Epochs = 0,
    [int]$InferWarmup = 0,
    [int]$InferSteps = 0,

    [float]$Alpha = 1.0,
    [float]$Beta = 1.0,
    [ValidateSet("weighted", "sentiment", "topic")]
    [string]$Selection = "weighted",

    [switch]$NoDaemon,
    [switch]$Aggregate
)

$ErrorActionPreference = "Stop"

function Parse-FloatList([string]$text) {
    $out = @()
    foreach ($p in ($text -split ",")) {
        $v = $p.Trim()
        if ($v.Length -eq 0) { continue }
        $out += [double]$v
    }
    if ($out.Count -eq 0) { throw "Invalid float list: $text" }
    return $out
}

function Parse-IntList([string]$text) {
    $out = @()
    foreach ($p in ($text -split ",")) {
        $v = $p.Trim()
        if ($v.Length -eq 0) { continue }
        $out += [int]$v
    }
    if ($out.Count -eq 0) { throw "Invalid int list: $text" }
    return $out
}

function Get-LatestRunSummaryPath {
    param(
        [string]$RunIdPrefix,
        [string]$OutputDir
    )

    $candidates = @(
        (Join-Path (Resolve-Path ".") "$OutputDir/JavaTorch/uit_vsfc_multitask"),
        (Join-Path (Resolve-Path ".") "examples/$OutputDir/JavaTorch/uit_vsfc_multitask"),
        (Join-Path (Resolve-Path ".") "$OutputDir/your_framework/uit_vsfc_multitask"),
        (Join-Path (Resolve-Path ".") "examples/$OutputDir/your_framework/uit_vsfc_multitask")
    )

    $all = @()
    foreach ($root in $candidates) {
        if (!(Test-Path $root)) {
            continue
        }
        $all += Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "${RunIdPrefix}*" } |
            Sort-Object LastWriteTime -Descending
    }

    if ($all.Count -eq 0) {
        return $null
    }

    foreach ($dir in $all) {
        $summary = Join-Path $dir.FullName "run_summary.csv"
        if (Test-Path $summary) {
            return $summary
        }
    }

    return $null
}

function Add-Phase3SummaryRow {
    param(
        [string]$SummaryCsv,
        [string]$RunSummaryPath,
        [hashtable]$Config
    )

    $r = Import-Csv $RunSummaryPath | Select-Object -First 1
    if ($null -eq $r) {
        return
    }

    $sent = [double]$r.test_sent_macro_f1
    $topic = [double]$r.test_topic_macro_f1
    $alpha = [double]$r.alpha
    $beta = [double]$r.beta
    $den = $alpha + $beta
    $weighted = if ($den -gt 0) { ($alpha / $den) * $sent + ($beta / $den) * $topic } else { ($sent + $topic) / 2.0 }

    $obj = [PSCustomObject]@{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        run_id = $r.run_id
        model = $r.model
        device = $r.device
        seed = $r.seed
        lr = $Config.lr
        min_lr = $Config.minLr
        warmup_epochs = $Config.warmup
        early_stopping_patience = $Config.patience
        early_stopping_min_delta = $Config.minDelta
        batch_size = $Config.batch
        max_len = $Config.maxLen
        epochs = $Config.epochs
        inference_warmup = $Config.inferWarmup
        inference_steps = $Config.inferSteps
        test_sent_macro_f1 = $r.test_sent_macro_f1
        test_topic_macro_f1 = $r.test_topic_macro_f1
        weighted_test_f1 = ([string]([math]::Round($weighted, 6)))
        test_joint_exact_match = $r.test_joint_exact_match
        total_train_time_ms = $r.total_train_time_ms
        inference_p50_ms = $r.inference_p50_ms
        inference_p95_ms = $r.inference_p95_ms
        inference_throughput_sps = $r.inference_throughput_sps
        best_epoch = $r.best_epoch
        best_dev_objective = $r.best_dev_objective
        run_summary_path = $RunSummaryPath
    }

    if (Test-Path $SummaryCsv) {
        $obj | Export-Csv -Path $SummaryCsv -NoTypeInformation -Append
    } else {
        $obj | Export-Csv -Path $SummaryCsv -NoTypeInformation
    }
}

$lrs = Parse-FloatList $LearningRates
$minLrs = Parse-FloatList $MinLearningRates
$warmups = Parse-IntList $WarmupEpochs
$patiences = Parse-IntList $EarlyStoppingPatience
$batches = Parse-IntList $BatchSizes
$maxLens = Parse-IntList $MaxLens

$sweepTag = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryOutDir = Join-Path $OutputDir "compare/uit_vsfc_multitask"
New-Item -ItemType Directory -Path $summaryOutDir -Force | Out-Null
$summaryCsv = Join-Path $summaryOutDir "phase3_sweep_${sweepTag}.csv"

$total = $lrs.Count * $minLrs.Count * $warmups.Count * $patiences.Count * $batches.Count * $maxLens.Count
$idx = 0

Write-Host "UIT-VSFC Phase 3 sweep"
Write-Host "Mode: $Mode | Device: $Device | Models: $Models | Seeds: $Seeds"
Write-Host "LRs: $($lrs -join ', ')"
Write-Host "MinLRs: $($minLrs -join ', ') | Schedule: $LrSchedule"
Write-Host "Warmup epochs: $($warmups -join ', ') | Patience: $($patiences -join ', ') | MinDelta: $EarlyStoppingMinDelta"
Write-Host "Batch sizes: $($batches -join ', ') | MaxLens: $($maxLens -join ', ')"
Write-Host "Manual overrides: epochs=$Epochs inferWarmup=$InferWarmup inferSteps=$InferSteps"
Write-Host "Sweep summary csv: $summaryCsv"

foreach ($lr in $lrs) {
    foreach ($minLr in $minLrs) {
        if ($minLr -gt $lr) {
            Write-Warning "Skip config minLr=$minLr > lr=$lr"
            continue
        }
        foreach ($wu in $warmups) {
            foreach ($pat in $patiences) {
                foreach ($bs in $batches) {
                    foreach ($mx in $maxLens) {
                        $idx++
                        Write-Host ""
                        Write-Host "[$idx/$total] Run config: lr=$lr minLr=$minLr warmup=$wu patience=$pat batch=$bs maxLen=$mx" -ForegroundColor Cyan

                        $runIdPrefix = "phase3_${sweepTag}_lr${lr}_mlr${minLr}_wu${wu}_pat${pat}_b${bs}_m${mx}"

                        if ($NoDaemon) {
                            & ./scripts/run-uit-vsfc-matrix.ps1 `
                                -Mode $Mode `
                                -Device $Device `
                                -Models $Models `
                                -Seeds $Seeds `
                                -DataDir $DataDir `
                                -OutputDir $OutputDir `
                                -Alpha $Alpha `
                                -Beta $Beta `
                                -Selection $Selection `
                                -LearningRate $lr `
                                -MinLearningRate $minLr `
                                -LrSchedule $LrSchedule `
                                -LrWarmupEpochs $wu `
                                -EarlyStoppingPatience $pat `
                                -EarlyStoppingMinDelta $EarlyStoppingMinDelta `
                                -BatchSize $bs `
                                -MaxLen $mx `
                                -Epochs $Epochs `
                                -InferWarmup $InferWarmup `
                                -InferSteps $InferSteps `
                                -RunIdPrefix $runIdPrefix `
                                -NoDaemon
                        } else {
                            & ./scripts/run-uit-vsfc-matrix.ps1 `
                                -Mode $Mode `
                                -Device $Device `
                                -Models $Models `
                                -Seeds $Seeds `
                                -DataDir $DataDir `
                                -OutputDir $OutputDir `
                                -Alpha $Alpha `
                                -Beta $Beta `
                                -Selection $Selection `
                                -LearningRate $lr `
                                -MinLearningRate $minLr `
                                -LrSchedule $LrSchedule `
                                -LrWarmupEpochs $wu `
                                -EarlyStoppingPatience $pat `
                                -EarlyStoppingMinDelta $EarlyStoppingMinDelta `
                                -BatchSize $bs `
                                -MaxLen $mx `
                                -Epochs $Epochs `
                                -InferWarmup $InferWarmup `
                                -InferSteps $InferSteps `
                                -RunIdPrefix $runIdPrefix
                        }
                        if ($LASTEXITCODE -ne 0) {
                            throw "Phase 3 sweep failed for config lr=$lr minLr=$minLr warmup=$wu patience=$pat batch=$bs maxLen=$mx"
                        }

                        $runSummaryPath = Get-LatestRunSummaryPath -RunIdPrefix $runIdPrefix -OutputDir $OutputDir
                        if ($null -eq $runSummaryPath) {
                            Write-Warning "Could not locate run_summary for prefix: $runIdPrefix"
                        } else {
                            Add-Phase3SummaryRow -SummaryCsv $summaryCsv -RunSummaryPath $runSummaryPath -Config @{
                                lr = $lr
                                minLr = $minLr
                                warmup = $wu
                                patience = $pat
                                minDelta = $EarlyStoppingMinDelta
                                batch = $bs
                                maxLen = $mx
                                epochs = $Epochs
                                inferWarmup = $InferWarmup
                                inferSteps = $InferSteps
                            }
                        }
                    }
                }
            }
        }
    }
}

if ($Aggregate) {
    Write-Host ""
    Write-Host "Aggregating comparison report..." -ForegroundColor Yellow
    $py = ".venv/Scripts/python.exe"
    if (!(Test-Path $py)) {
        $py = "python"
    }
    & $py scripts/aggregate_uit_vsfc_compare.py --resultsDir benchmark/results --latestOnly --dropMissingMetrics
}

Write-Host ""
if (Test-Path $summaryCsv) {
    Write-Host "Top 5 by weighted_test_f1:" -ForegroundColor Yellow
    Import-Csv $summaryCsv |
        Sort-Object {[double]$_.weighted_test_f1} -Descending |
        Select-Object -First 5 run_id,device,model,weighted_test_f1,total_train_time_ms,inference_p50_ms,inference_throughput_sps |
        Format-Table -AutoSize
}
Write-Host "Sweep summary CSV: $summaryCsv"
Write-Host "Phase 3 sweep completed." -ForegroundColor Green
