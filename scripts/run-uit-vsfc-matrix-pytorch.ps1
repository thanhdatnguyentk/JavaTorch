param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "full",

    [ValidateSet("cpu", "gpu", "both")]
    [string]$Device = "gpu",

    [ValidateSet("all", "lstm", "transformer")]
    [string]$Models = "all",

    [string]$Seeds = "42",
    [string]$DataDir = "examples/data/uit-vsfc",
    [string]$OutputDir = "benchmark/results",
    [float]$Alpha = 1.0,
    [float]$Beta = 1.0,
    [ValidateSet("weighted", "sentiment", "topic")]
    [string]$Selection = "weighted"
)

$ErrorActionPreference = "Stop"

function Get-RunConfig {
    if ($Mode -eq "quick") {
        return @{ epochs = 1; batch = 256; maxLen = 48; inferWarmup = 5; inferSteps = 20 }
    }
    return @{ epochs = 50; batch = 256; maxLen = 64; inferWarmup = 10; inferSteps = 100 }
}

function Invoke-UitPyTorchRun {
    param(
        [string]$Model,
        [string]$DeviceName,
        [long]$SeedValue
    )

    $cfg = Get-RunConfig
    $runId = "uit_vsfc_${Model}_${DeviceName}_seed$($SeedValue)_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

    $pythonExe = ".venv/Scripts/python.exe"
    if (!(Test-Path $pythonExe)) {
        $pythonExe = "python"
    }

    Write-Host ""
    Write-Host "=== UIT-VSFC PyTorch | model=$Model device=$DeviceName seed=$SeedValue mode=$Mode ===" -ForegroundColor Cyan
    Write-Host "$pythonExe scripts/benchmark_uit_vsfc_multitask_pytorch.py --dataDir $DataDir --epochs $($cfg.epochs) --batchSize $($cfg.batch) --maxLen $($cfg.maxLen) --model $Model --device $DeviceName --seed $SeedValue --alpha $Alpha --beta $Beta --selection $Selection --inferWarmup $($cfg.inferWarmup) --inferSteps $($cfg.inferSteps) --outputDir $OutputDir --runId $runId" -ForegroundColor DarkGray

    & $pythonExe "scripts/benchmark_uit_vsfc_multitask_pytorch.py" "--dataDir" $DataDir "--epochs" "$($cfg.epochs)" "--batchSize" "$($cfg.batch)" "--maxLen" "$($cfg.maxLen)" "--model" $Model "--device" $DeviceName "--seed" "$SeedValue" "--alpha" "$Alpha" "--beta" "$Beta" "--selection" $Selection "--inferWarmup" "$($cfg.inferWarmup)" "--inferSteps" "$($cfg.inferSteps)" "--outputDir" $OutputDir "--runId" $runId
    if ($LASTEXITCODE -ne 0) {
        throw "UIT-VSFC PyTorch run failed: model=$Model device=$DeviceName seed=$SeedValue"
    }
}

$deviceSet = if ($Device -eq "both") { @("cpu", "gpu") } else { @($Device) }
$modelSet = if ($Models -eq "all") { @("lstm", "transformer") } else { @($Models) }
$seedSet = @()
foreach ($s in ($Seeds -split ",")) {
    $trimmed = $s.Trim()
    if ($trimmed.Length -eq 0) {
        continue
    }
    $seedSet += [long]$trimmed
}

if ($seedSet.Count -eq 0) {
    throw "No valid seeds provided. Example: -Seeds '42,1337'"
}

Write-Host "UIT-VSFC PyTorch benchmark matrix"
Write-Host "Mode: $Mode"
Write-Host "Models: $($modelSet -join ', ')"
Write-Host "Devices: $($deviceSet -join ', ')"
Write-Host "Seeds: $($seedSet -join ', ')"
Write-Host "DataDir: $DataDir"
Write-Host "OutputDir: $OutputDir"
Write-Host "Selection: $Selection"
Write-Host "Alpha/Beta: $Alpha/$Beta"

foreach ($m in $modelSet) {
    foreach ($d in $deviceSet) {
        foreach ($s in $seedSet) {
            Invoke-UitPyTorchRun -Model $m -DeviceName $d -SeedValue $s
        }
    }
}

Write-Host ""
Write-Host "UIT-VSFC PyTorch matrix completed. Artifacts are in: $OutputDir" -ForegroundColor Green
