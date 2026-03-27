param(
    [string]$DatasetDir = "data/anime_faces",
    [int]$Epochs = 30,
    [int]$BatchSize = 64,
    [int]$MaxImages = -1,
    [int]$LatentDim = 100,
    [int]$SampleCount = 8
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Building core classes..."
& .\gradlew.bat :core:classes --no-daemon
if ($LASTEXITCODE -ne 0) {
    throw "Gradle build failed with exit code $LASTEXITCODE"
}

if (!(Test-Path $DatasetDir)) {
    throw "Dataset folder not found: $DatasetDir"
}

$cp = "core/build/classes/java/main;core/build/resources/main;lib/*"

Write-Host "[2/3] Training GAN on: $DatasetDir"
java --add-modules jdk.incubator.vector -cp $cp com.user.nn.examples.TrainGANAnime $DatasetDir $Epochs $BatchSize $MaxImages
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

Write-Host "[3/3] Generating sample images..."
java --add-modules jdk.incubator.vector -cp $cp com.user.nn.examples.AnimeGenerator generated/anime_gan/gan_anime_generator.bin $LatentDim $SampleCount generated/anime_gan/samples
if ($LASTEXITCODE -ne 0) {
    throw "Image generation failed with exit code $LASTEXITCODE"
}

Write-Host "Done. Check generated/anime_gan/ and generated/anime_gan/samples/"
