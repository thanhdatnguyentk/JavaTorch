param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [ValidateSet("lstm", "transformer")]
    [string]$Model = "lstm",

    [long]$Seed = 42,
    [string]$DataDir = "examples/data/uit-vsfc",
    [string]$OutputDir = "benchmark/results",

    [switch]$SkipGpuSmoke,
    [switch]$NoDaemon
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Get-RunConfig {
    if ($Mode -eq "quick") {
        return @{ epochs = 1; batch = 256; maxLen = 48; inferWarmup = 5; inferSteps = 20 }
    }
    return @{ epochs = 50; batch = 256; maxLen = 64; inferWarmup = 10; inferSteps = 100 }
}

function Invoke-Gradle {
    param([string[]]$TaskArgs)

    Write-Host ".\gradlew.bat $($TaskArgs -join ' ')" -ForegroundColor DarkGray
    & .\gradlew.bat @TaskArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Gradle command failed with exit code $LASTEXITCODE"
    }
}

function Test-NvidiaSmi {
    try {
        $null = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        return $true
    }
    catch {
        return $false
    }
}

function Invoke-UitRun {
    param(
        [ValidateSet("gpu", "cpu")]
        [string]$Device,
        [switch]$SampleTelemetry
    )

    $cfg = Get-RunConfig
    $runId = "phase2_${Model}_${Device}_seed${Seed}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    $runDir = Join-Path (Resolve-Path ".") "examples/benchmark/results/JavaTorch/uit_vsfc_multitask/$runId"

    $argLine = "--dataDir=$DataDir --epochs=$($cfg.epochs) --batchSize=$($cfg.batch) --maxLen=$($cfg.maxLen) --model=$Model --device=$Device --seed=$Seed --alpha=1 --beta=1 --learningRate=0.001 --minLearningRate=0.0001 --lrSchedule=none --lrWarmupEpochs=0 --earlyStoppingPatience=0 --earlyStoppingMinDelta=0 --selection=weighted --inferWarmup=$($cfg.inferWarmup) --inferSteps=$($cfg.inferSteps) --outputDir=$OutputDir --runId=$runId"

    $gradleArgs = @(
        "-PmainClass=com.user.nn.examples.TrainUitVsfcMultitask",
        ":examples:run",
        "--args=$argLine",
        "--console=plain"
    )

    if ($NoDaemon) {
        $gradleArgs += "--no-daemon"
    }

    Write-Host "`n=== UIT-VSFC | device=$Device model=$Model seed=$Seed mode=$Mode ===" -ForegroundColor Yellow
    Write-Host ".\gradlew.bat $($gradleArgs -join ' ')" -ForegroundColor DarkGray

    $telemetryRecords = @()
    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    $combinedText = ""
    $cudaInitDetected = $false

    try {
        $procArgLine = ($gradleArgs | ForEach-Object {
            if ($_ -match "\s") {
                '"' + ($_ -replace '"', '\"') + '"'
            }
            else {
                $_
            }
        }) -join " "

        $proc = Start-Process -FilePath ".\gradlew.bat" -ArgumentList $procArgLine -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr

        while (-not $proc.HasExited) {
            if ($SampleTelemetry -and (Test-NvidiaSmi)) {
                $timestamp = Get-Date -Format "o"
                $lines = & nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>$null
                if ($lines) {
                    foreach ($line in $lines) {
                        if ($line -match "java\.exe") {
                            $telemetryRecords += "$timestamp,$line"
                        }
                    }
                }
            }
            Start-Sleep -Seconds 2
            $proc.Refresh()
        }

        $outText = if (Test-Path $stdout) { Get-Content -Path $stdout -Raw } else { "" }
        $errText = if (Test-Path $stderr) { Get-Content -Path $stderr -Raw } else { "" }
        $combinedText = "$outText`n$errText"
        $cudaInitDetected = $combinedText -match "\[CUDAOps\]\s+CUDA initialization complete\."

        if ($outText.Trim().Length -gt 0) {
            Write-Host $outText
        }
        if ($errText.Trim().Length -gt 0) {
            Write-Host $errText
        }

        $exitCode = $proc.ExitCode
        if ($null -eq $exitCode) {
            if ($combinedText -match "BUILD SUCCESSFUL") {
                $exitCode = 0
            }
            else {
                $exitCode = 1
            }
        }

        if ($exitCode -ne 0) {
            throw "Run failed for device=$Device (exit=$exitCode)."
        }
    }
    finally {
        if (-not (Test-Path $runDir)) {
            New-Item -Path $runDir -ItemType Directory -Force | Out-Null
        }
        $logPath = Join-Path $runDir "phase2_console.log"
        $combinedText | Set-Content -Path $logPath

        Remove-Item -Path $stdout -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $stderr -Force -ErrorAction SilentlyContinue
    }

    $summaryPath = Join-Path $runDir "run_summary.csv"
    if (-not (Test-Path $summaryPath)) {
        throw "Run summary not found: $summaryPath"
    }

    return [PSCustomObject]@{
        Device = $Device
        RunId = $runId
        RunDir = $runDir
        SummaryPath = $summaryPath
        TelemetryRecords = $telemetryRecords
        CudaInitDetected = $cudaInitDetected
    }
}

function Get-SummaryMetrics {
    param([string]$Path)

    $r = Import-Csv $Path
    return [PSCustomObject]@{
        run_id = $r.run_id
        device = $r.device
        total_train_time_s = [math]::Round(([double]$r.total_train_time_ms) / 1000.0, 2)
        inference_p50_ms = [math]::Round([double]$r.inference_p50_ms, 2)
        inference_p95_ms = [math]::Round([double]$r.inference_p95_ms, 2)
        inference_throughput_sps = [math]::Round([double]$r.inference_throughput_sps, 2)
        inference_avg_h2d_ms = [math]::Round([double]$r.inference_avg_h2d_ms, 4)
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"
Write-Host "Phase 2 GPU verification"
Write-Host "Mode: $Mode | Model: $Model | Seed: $Seed"

Write-Section "Preflight"
$gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Write-Host "NVIDIA: $gpuInfo"
$driverTop = & nvidia-smi | Select-Object -First 3
$driverTop | ForEach-Object { Write-Host $_ }
$javaArch = cmd /c "java -XshowSettings:properties -version 2>&1" | Select-String "os.arch|java.version"
$javaArch | ForEach-Object { Write-Host $_ }

if (-not $SkipGpuSmoke) {
    Write-Section "GPU smoke"
    Invoke-Gradle -TaskArgs @(
        ":core:gpuSmoke",
        "--console=plain",
        "--no-daemon"
    )
}

Write-Section "Workload runs"
$gpuRun = Invoke-UitRun -Device "gpu" -SampleTelemetry
$cpuRun = Invoke-UitRun -Device "cpu"

Write-Section "Result summary"
$gpuMetrics = Get-SummaryMetrics -Path $gpuRun.SummaryPath
$cpuMetrics = Get-SummaryMetrics -Path $cpuRun.SummaryPath

Write-Host "GPU run: $($gpuRun.RunId)"
Write-Host "CPU run: $($cpuRun.RunId)"

$summary = @(
    [PSCustomObject]@{ metric = "total_train_time_s"; gpu = $gpuMetrics.total_train_time_s; cpu = $cpuMetrics.total_train_time_s }
    [PSCustomObject]@{ metric = "inference_p50_ms"; gpu = $gpuMetrics.inference_p50_ms; cpu = $cpuMetrics.inference_p50_ms }
    [PSCustomObject]@{ metric = "inference_p95_ms"; gpu = $gpuMetrics.inference_p95_ms; cpu = $cpuMetrics.inference_p95_ms }
    [PSCustomObject]@{ metric = "inference_throughput_sps"; gpu = $gpuMetrics.inference_throughput_sps; cpu = $cpuMetrics.inference_throughput_sps }
    [PSCustomObject]@{ metric = "inference_avg_h2d_ms"; gpu = $gpuMetrics.inference_avg_h2d_ms; cpu = $cpuMetrics.inference_avg_h2d_ms }
)
$summary | Format-Table -AutoSize

$gpuCudaInitOk = $gpuRun.CudaInitDetected
$telemetryCount = @($gpuRun.TelemetryRecords).Count

$telemetryPath = Join-Path $gpuRun.RunDir "nvidia_compute_samples.csv"
if ($telemetryCount -gt 0) {
    "timestamp,pid,process_name,used_gpu_memory" | Set-Content -Path $telemetryPath
    foreach ($rec in $gpuRun.TelemetryRecords) {
        Add-Content -Path $telemetryPath -Value $rec
    }
    Write-Host "Telemetry samples captured: $telemetryCount"
    Write-Host "Telemetry file: $telemetryPath"
}
else {
    Write-Host "Telemetry samples captured: 0" -ForegroundColor Yellow
}

Write-Section "Phase 2 verdict"
if ($gpuCudaInitOk -and $telemetryCount -gt 0) {
    Write-Host "PASS: GPU compute path verified via workload execution and NVIDIA compute-process telemetry." -ForegroundColor Green
}
else {
    Write-Host "WARN: GPU evidence incomplete. Re-run and inspect CUDA logs + nvidia-smi samples." -ForegroundColor Yellow
}

Write-Host "`nArtifacts:"
Write-Host "- GPU: $($gpuRun.RunDir)"
Write-Host "- CPU: $($cpuRun.RunDir)"