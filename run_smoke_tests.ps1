$ErrorActionPreference = "Continue"

$examplesDir = "src\com\user\nn\examples"
$classes = Get-ChildItem -Path $examplesDir -Filter "*.java" | Where-Object { $_.Name -ne "SmokeTest.java" } | Select-Object -ExpandProperty BaseName

$logFile = "smoke_test_results.log"
Clear-Content -Path $logFile -ErrorAction SilentlyContinue

Write-Output "Starting Smoke Tests for $($classes.Count) classes sequentially..." | Tee-Object -FilePath $logFile -Append

foreach ($className in $classes) {
    $fullClass = "com.user.nn.examples.$className"
    Write-Output "`n==================================================" | Tee-Object -FilePath $logFile -Append
    Write-Output "Testing: $fullClass" | Tee-Object -FilePath $logFile -Append
    
    # Run synchronously
    $process = Start-Process -FilePath ".\gradlew.bat" -ArgumentList ":examples:run", "-DsmokeTest=true", "-PmainClass=$fullClass", "--no-daemon" -PassThru -NoNewWindow -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Output "[$className] PASS (Exit Code: 0)" | Tee-Object -FilePath $logFile -Append
    } else {
        Write-Output "[$className] FAILED (Exit Code: $($process.ExitCode))" | Tee-Object -FilePath $logFile -Append
    }
}

Write-Output "`nSmoke Test Complete." | Tee-Object -FilePath $logFile -Append
