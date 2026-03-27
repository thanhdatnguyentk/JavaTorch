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
    [float]$LearningRate = 0.001,
    [float]$MinLearningRate = 0.0001,
    [ValidateSet("none", "cosine")]
    [string]$LrSchedule = "none",
    [int]$LrWarmupEpochs = 0,
    [int]$EarlyStoppingPatience = 0,
    [float]$EarlyStoppingMinDelta = 0.0,
    [ValidateSet("weighted", "sentiment", "topic")]
    [string]$Selection = "weighted",

    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [int]$MaxLen = 0,
    [int]$InferWarmup = 0,
    [int]$InferSteps = 0,
    [string]$RunIdPrefix = "uit_vsfc",
    [switch]$NoDaemon
)

$ErrorActionPreference = "Stop"

function Get-RunConfig {
    if ($Mode -eq "quick") {
        $cfg = @{ epochs = 1; batch = 256; maxLen = 48; inferWarmup = 5; inferSteps = 20 }
    } else {
        $cfg = @{ epochs = 50; batch = 256; maxLen = 64; inferWarmup = 10; inferSteps = 100 }
    }

    if ($Epochs -gt 0) { $cfg.epochs = $Epochs }
    if ($BatchSize -gt 0) { $cfg.batch = $BatchSize }
    if ($MaxLen -gt 0) { $cfg.maxLen = $MaxLen }
    if ($InferWarmup -gt 0) { $cfg.inferWarmup = $InferWarmup }
    if ($InferSteps -gt 0) { $cfg.inferSteps = $InferSteps }

    return $cfg
}

function Invoke-UitRun {
    param(
        [string]$Model,
        [string]$DeviceName,
        [long]$SeedValue
    )

    $cfg = Get-RunConfig
    $runId = "${RunIdPrefix}_${Model}_$($DeviceName)_seed$($SeedValue)_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

    $argLine = "--dataDir=$DataDir --epochs=$($cfg.epochs) --batchSize=$($cfg.batch) --maxLen=$($cfg.maxLen) --model=$Model --device=$DeviceName --seed=$SeedValue --alpha=$Alpha --beta=$Beta --learningRate=$LearningRate --minLearningRate=$MinLearningRate --lrSchedule=$LrSchedule --lrWarmupEpochs=$LrWarmupEpochs --earlyStoppingPatience=$EarlyStoppingPatience --earlyStoppingMinDelta=$EarlyStoppingMinDelta --selection=$Selection --inferWarmup=$($cfg.inferWarmup) --inferSteps=$($cfg.inferSteps) --outputDir=$OutputDir --runId=$runId"

    $gradleArgs = @(
        "-PmainClass=com.user.nn.examples.TrainUitVsfcMultitask",
        ":examples:run",
        "--args=$argLine",
        "--console=plain"
    )

    if ($NoDaemon) {
        $gradleArgs += "--no-daemon"
    }

    Write-Host ""
    Write-Host "=== UIT-VSFC | model=$Model device=$DeviceName seed=$SeedValue mode=$Mode ===" -ForegroundColor Cyan
    Write-Host ".\gradlew.bat $($gradleArgs -join ' ')" -ForegroundColor DarkGray

    & .\gradlew.bat @gradleArgs
    if ($LASTEXITCODE -ne 0) {
        throw "UIT-VSFC run failed: model=$Model device=$DeviceName seed=$SeedValue"
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

Write-Host "UIT-VSFC benchmark matrix"
Write-Host "Mode: $Mode"
Write-Host "Models: $($modelSet -join ', ')"
Write-Host "Devices: $($deviceSet -join ', ')"
Write-Host "Seeds: $($seedSet -join ', ')"
Write-Host "DataDir: $DataDir"
Write-Host "OutputDir: $OutputDir"
Write-Host "Selection: $Selection"
Write-Host "Alpha/Beta: $Alpha/$Beta"
Write-Host "LR config: lr=$LearningRate minLr=$MinLearningRate schedule=$LrSchedule warmup=$LrWarmupEpochs"
Write-Host "Early stopping: patience=$EarlyStoppingPatience minDelta=$EarlyStoppingMinDelta"
Write-Host "Overrides: epochs=$Epochs batch=$BatchSize maxLen=$MaxLen inferWarmup=$InferWarmup inferSteps=$InferSteps"
Write-Host "RunId prefix: $RunIdPrefix"
Write-Host "Gradle daemon: $(if ($NoDaemon) { 'disabled' } else { 'enabled' })"

foreach ($m in $modelSet) {
    foreach ($d in $deviceSet) {
        foreach ($s in $seedSet) {
            Invoke-UitRun -Model $m -DeviceName $d -SeedValue $s
        }
    }
}

Write-Host ""
Write-Host "UIT-VSFC matrix completed. Artifacts are in: $OutputDir" -ForegroundColor Green
