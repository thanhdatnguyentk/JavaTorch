param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [int]$ExampleTimeoutSec = 60,

    [switch]$SkipExamples,

    [switch]$IncludeDl4jBenchmarks
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Invoke-Gradle {
    param([string[]]$Tasks)

    $gradleArgs = @($Tasks + "--no-daemon")
    if (Test-Path "./gradlew.bat") {
        Write-Host "> .\\gradlew.bat $($gradleArgs -join ' ')"
        & ./gradlew.bat @gradleArgs
    }
    elseif (Test-Path "./gradlew") {
        Write-Host "> ./gradlew $($gradleArgs -join ' ')"
        & ./gradlew @gradleArgs
    }
    else {
        throw "Gradle wrapper not found. Expected gradlew or gradlew.bat in repository root."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Gradle command failed with exit code $LASTEXITCODE"
    }
}

function Get-ExampleMainClasses {
    $exampleDir = Join-Path (Get-Location) "src/com/user/nn/examples"
    if (!(Test-Path $exampleDir)) {
        throw "Examples directory not found: $exampleDir"
    }

    $all = Get-ChildItem -Path $exampleDir -Filter "*.java" | Sort-Object Name
    $mainClasses = @()

    foreach ($file in $all) {
        $content = Get-Content -Path $file.FullName -Raw
        if ($content -match "public\s+static\s+void\s+main\s*\(") {
            $mainClasses += "com.user.nn.examples.$($file.BaseName)"
        }
    }

    return $mainClasses
}

function Test-MainClassCompiled {
    param([string]$MainClass)

    $name = $MainClass.Split('.')[-1]
    $classCandidates = @(
        (Join-Path (Get-Location) "core/build/classes/java/main/com/user/nn/examples/$name.class"),
        (Join-Path (Get-Location) "examples/build/classes/java/main/com/user/nn/examples/$name.class")
    )
    return ($classCandidates | Where-Object { Test-Path $_ }).Count -gt 0
}

function Invoke-ExampleSmoke {
    param(
        [string]$MainClass,
        [string]$Classpath,
        [int]$TimeoutSec
    )

    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()

    try {
        $args = @(
            "--add-modules", "jdk.incubator.vector",
            "-cp", $Classpath,
            $MainClass
        )

        $proc = Start-Process -FilePath "java" -ArgumentList $args -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr

        $timedOut = $false
        try {
            Wait-Process -Id $proc.Id -Timeout $TimeoutSec
        }
        catch {
            $timedOut = $true
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }

        if (-not $timedOut) {
            if (-not $proc.HasExited) {
                $proc.WaitForExit()
            }
            $proc.Refresh()
        }

        $outText = ""
        $errText = ""

        if (Test-Path $stdout) {
            $outText = (Get-Content -Path $stdout -Raw)
        }
        if (Test-Path $stderr) {
            $errText = (Get-Content -Path $stderr -Raw)
        }

        $combined = "$errText`n$outText"
        $fatalPattern = "Exception in thread|Unresolved compilation problem|ClassNotFoundException|NoClassDefFoundError|UnsatisfiedLinkError"

        if ($timedOut) {
            if ($combined -match $fatalPattern) {
                return [PSCustomObject]@{
                    MainClass = $MainClass
                    Status = "FAIL"
                    Detail = "Timed out and fatal error was detected"
                    Tail = ($combined -split "`n" | Select-Object -Last 20) -join "`n"
                }
            }

            return [PSCustomObject]@{
                MainClass = $MainClass
                Status = "PASS"
                Detail = "No immediate fatal error within ${TimeoutSec}s"
                Tail = ($combined -split "`n" | Select-Object -Last 10) -join "`n"
            }
        }

        $exitCode = $proc.ExitCode
        if ($null -eq $exitCode) {
            $exitCode = 0
        }

        if ($exitCode -eq 0) {
            return [PSCustomObject]@{
                MainClass = $MainClass
                Status = "PASS"
                Detail = "Exited with code 0"
                Tail = ($combined -split "`n" | Select-Object -Last 10) -join "`n"
            }
        }

        return [PSCustomObject]@{
            MainClass = $MainClass
            Status = "FAIL"
            Detail = "Exited with code $exitCode"
            Tail = ($combined -split "`n" | Select-Object -Last 20) -join "`n"
        }
    }
    finally {
        Remove-Item -Path $stdout -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $stderr -Force -ErrorAction SilentlyContinue
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"
Write-Host "Mode: $Mode"

Write-Section "Build and tests"
Invoke-Gradle -Tasks @(
    ":core:build",
    ":core:test",
    ":tests:test",
    ":examples:classes"
)

Write-Section "Discover and validate examples"
$mainClasses = Get-ExampleMainClasses

if (-not $IncludeDl4jBenchmarks) {
    $mainClasses = $mainClasses | Where-Object { $_ -notmatch "^com\.user\.nn\.examples\.BenchmarkDl4j" }
}

if ($mainClasses.Count -eq 0) {
    throw "No examples with main() were discovered in src/com/user/nn/examples"
}

$missingCompiled = @()
foreach ($c in $mainClasses) {
    if (-not (Test-MainClassCompiled -MainClass $c)) {
        $missingCompiled += $c
    }
}

if ($missingCompiled.Count -gt 0) {
    Write-Host "Missing compiled classes:" -ForegroundColor Red
    $missingCompiled | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 2
}

Write-Host "Discovered $($mainClasses.Count) example entrypoints."

if ($SkipExamples) {
    Write-Host "Skipping runtime smoke tests because -SkipExamples was specified." -ForegroundColor Yellow
    exit 0
}

$sep = [System.IO.Path]::PathSeparator
$cp = @(
    (Join-Path $repoRoot "core/build/classes/java/main"),
    (Join-Path $repoRoot "examples/build/classes/java/main"),
    (Join-Path $repoRoot "lib/*")
) -join $sep

if ($Mode -eq "quick") {
    $preferred = @(
        "com.user.nn.examples.TrainIris",
        "com.user.nn.examples.TrainSentiment",
        "com.user.nn.examples.ObjectDetectionDemo"
    )
    $runList = $preferred | Where-Object { $mainClasses -contains $_ }
    if ($runList.Count -eq 0) {
        $runList = $mainClasses | Select-Object -First 3
    }
}
else {
    $runList = $mainClasses
}

Write-Section "Runtime smoke tests"
Write-Host "Running $($runList.Count) example(s) with timeout ${ExampleTimeoutSec}s each."

$results = @()
foreach ($mc in $runList) {
    Write-Host "`n[SMOKE] $mc"
    $res = Invoke-ExampleSmoke -MainClass $mc -Classpath $cp -TimeoutSec $ExampleTimeoutSec
    $results += $res

    if ($res.Status -eq "PASS") {
        Write-Host "PASS: $($res.Detail)" -ForegroundColor Green
    }
    else {
        Write-Host "FAIL: $($res.Detail)" -ForegroundColor Red
        if ($res.Tail) {
            Write-Host $res.Tail
        }
    }
}

$failed = @($results | Where-Object { $_.Status -eq "FAIL" })

Write-Section "Summary"
$results | Select-Object MainClass, Status, Detail | Format-Table -AutoSize

if (@($failed).Count -gt 0) {
    Write-Host "Smoke test failed for $($failed.Count) example(s)." -ForegroundColor Red
    exit 2
}

Write-Host "All CI checks passed." -ForegroundColor Green
exit 0
