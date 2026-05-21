$ErrorActionPreference = "Stop"

$models = @(
    "runTrainFashionMNIST",
    "runTrainCifar10",
    "runTrainResNet",
    "runTrainViTCifar10",
    "runTrainSentiment",
    "exampleUitVsfc",
    "runTrainUitVsfcMultitask"
)

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$summaryCsv = "benchmark_comparison_summary.csv"
if (-not (Test-Path $summaryCsv)) {
    "Timestamp,Framework,Model,Device,E2E_Time_Seconds,Peak_RAM_MB,Peak_VRAM_MB" | Out-File $summaryCsv -Encoding UTF8
}

Write-Host "Starting full-scale training marathon..."

foreach ($model in $models) {
    Write-Host "`n==============================================="
    Write-Host ">>> Starting $model"
    Write-Host "==============================================="
    
    $logFile = "logs\$model.log"
    
    # Run Gradle task. Piping to Tee-Object to capture both file and console if needed, 
    # but here we just redirect to file to avoid clogging up stdout and also to have a record.
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

    # Run Gradle task with forceEpochs=20
    ./gradlew :examples:$model -DforceEpochs=20 | Tee-Object -FilePath $logFile

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
    "$ts,JavaTorch_Marathon,$model,gpu,$e2eTime,$peakRam,$peakVram" | Out-File $summaryCsv -Encoding UTF8 -Append
    
    Write-Host "<<< Finished $model"
    
    # Force garbage collection between runs to help OS release memory
    [System.GC]::Collect()
    Start-Sleep -Seconds 2
}

Write-Host "`nAll models have been executed!"
