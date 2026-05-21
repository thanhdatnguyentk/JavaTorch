param(
    [string]$Device = "gpu",
    [int]$Epochs = 8
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Optional argument parsing (like -Device gpu)


$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       PyTorch Benchmark Suite - $timestamp" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Device    : $Device"
Write-Host "  Epochs    : $Epochs"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$summaryCsv = "..\benchmark_comparison_summary.csv"
if (-not (Test-Path $summaryCsv)) {
    "Timestamp,Framework,Model,Device,E2E_Time_Seconds,Peak_RAM_MB,Peak_VRAM_MB" | Out-File $summaryCsv -Encoding UTF8
}

$benchmarks = @(
    @{
        Name   = "Sentiment (RT-Polarity)"
        Script = "sentiment_rtpolarity.py"
        Epochs = $Epochs
    },
    @{
        Name   = "ResNet (CIFAR-10)"
        Script = "resnet_cifar10.py"
        Epochs = $Epochs
    },
    @{
        Name   = "LeNet (FashionMNIST)"
        Script = "fashion_mnist.py"
        Epochs = $Epochs
    },
    @{
        Name   = "ViT (CIFAR-10)"
        Script = "vit_cifar10.py"
        Epochs = 20
    },
    @{
        Name   = "Iris (MLP)"
        Script = "iris_mlp.py"
        Epochs = 500
    },
    @{
        Name   = "LeNet (MNIST)"
        Script = "lenet_mnist.py"
        Epochs = 15
    },
    @{
        Name   = "UIT-VSFC (Multitask LSTM)"
        Script = "uit_vsfc_multitask.py"
        Epochs = 10
    }
)

foreach ($b in $benchmarks) {
    Write-Host "`n==============================================="
    Write-Host ">>> Starting $($b.Name) in PyTorch"
    Write-Host "==============================================="
    
    $scriptName = $b.Script
    $epochsToRun = $b.Epochs
    $logFile = "logs\$scriptName.log"
    
    Write-Host "Running: py -3.10 $scriptName --device $Device --epochs $epochsToRun"
    
    $stopFile = "$PWD\stop_monitor.tmp"
    if (Test-Path $stopFile) { Remove-Item $stopFile }
    $job = Start-Job -ScriptBlock {
        param($stopFile)
        $maxRam = 0
        $maxVram = 0
        while (-not (Test-Path $stopFile)) {
            try {
                $procs = Get-Process -Name "python" -ErrorAction SilentlyContinue
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

    # Run the Python script and log it
    $cmd = "py -3.10 $scriptName --device $Device --epochs $epochsToRun"
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Invoke-Expression "$cmd 2>&1 | Tee-Object -FilePath $logFile"
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldEap
    
    $sw.Stop()
    Out-File -FilePath $stopFile -InputObject "stop"
    Wait-Job $job | Out-Null
    $jobResult = Receive-Job $job
    Remove-Job $job
    if (Test-Path $stopFile) { Remove-Item $stopFile }

    if ($exitCode -ne 0) {
        Write-Error "Benchmark script $scriptName failed with exit code $exitCode"
        break
    }

    $peakRam = [math]::Round($jobResult.MaxRam, 2)
    $peakVram = $jobResult.MaxVram
    $e2eTime = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts,PyTorch,$($b.Name),$Device,$e2eTime,$peakRam,$peakVram" | Out-File $summaryCsv -Encoding UTF8 -Append

    Write-Host "<<< Finished $($b.Name)"
    Start-Sleep -Seconds 2
}

Write-Host "`nAll PyTorch benchmarks have been executed!"
