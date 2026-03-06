# Download JCuda dependencies

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $root

$libDir = Join-Path $root 'lib'
if (!(Test-Path $libDir)) {
    New-Item -ItemType Directory -Path $libDir | Out-Null
}

$jcudaVersion = "12.0.0" 
# Or we can use the latest stable version under org.jcuda
# Maven Central URLs
$baseUrl = "https://repo1.maven.org/maven2/org/jcuda"

$filesToDownload = @(
    @{ Name = "jcuda-$jcudaVersion.jar"; Url = "$baseUrl/jcuda/$jcudaVersion/jcuda-$jcudaVersion.jar" },
    @{ Name = "jcuda-natives-windows-x86_64-$jcudaVersion.jar"; Url = "$baseUrl/jcuda-natives/$jcudaVersion/jcuda-natives-$jcudaVersion-windows-x86_64.jar" },
    @{ Name = "jcublas-$jcudaVersion.jar"; Url = "$baseUrl/jcublas/$jcudaVersion/jcublas-$jcudaVersion.jar" },
    @{ Name = "jcublas-natives-windows-x86_64-$jcudaVersion.jar"; Url = "$baseUrl/jcublas-natives/$jcudaVersion/jcublas-natives-$jcudaVersion-windows-x86_64.jar" },
    @{ Name = "jcudnn-$jcudaVersion.jar"; Url = "$baseUrl/jcudnn/$jcudaVersion/jcudnn-$jcudaVersion.jar" },
    @{ Name = "jcudnn-natives-windows-x86_64-$jcudaVersion.jar"; Url = "$baseUrl/jcudnn-natives/$jcudaVersion/jcudnn-natives-$jcudaVersion-windows-x86_64.jar" }
)

Write-Host "Downloading JCuda libraries to $libDir..."

foreach ($file in $filesToDownload) {
    $dest = Join-Path $libDir $file.Name
    if (!(Test-Path $dest)) {
        Write-Host "Downloading $($file.Name)..."
        try {
            Invoke-WebRequest -Uri $file.Url -OutFile $dest
            Write-Host "  Success."
        } catch {
            Write-Error "Failed to download $($file.Name): $_"
        }
    } else {
        Write-Host "$($file.Name) already exists."
    }
}

Write-Host "Done downloading dependencies."
