$ErrorActionPreference = "Stop"

$models = @(
    # "runTrainFashionMNIST",
    # "runTrainCifar10",
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

Write-Host "Starting full-scale training marathon..."

foreach ($model in $models) {
    Write-Host "`n==============================================="
    Write-Host ">>> Starting $model"
    Write-Host "==============================================="
    
    $logFile = "logs\$model.log"
    
    # Run Gradle task. Piping to Tee-Object to capture both file and console if needed, 
    # but here we just redirect to file to avoid clogging up stdout and also to have a record.
    # Run Gradle task with forceEpochs=20
    ./gradlew :examples:$model -DforceEpochs=20 | Tee-Object -FilePath $logFile
    
    Write-Host "<<< Finished $model"
    
    # Force garbage collection between runs to help OS release memory
    [System.GC]::Collect()
    Start-Sleep -Seconds 2
}

Write-Host "`nAll models have been executed!"
