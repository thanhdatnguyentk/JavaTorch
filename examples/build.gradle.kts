plugins {
    application
}

dependencies {
    implementation(project(":core"))
}

application {
    val cliMain = findProperty("mainClass") as String?
    mainClass.set(cliMain ?: "com.user.nn.examples.TrainYOLOCoco")
}

tasks.withType<JavaExec>().configureEach {
    // core uses the Vector API; run tasks must enable incubator module at runtime.
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "6g"
    javaLauncher.set(javaToolchains.launcherFor {
        languageVersion.set(JavaLanguageVersion.of(21))
    })
}

tasks.register<Copy>("ensureKernelsPtx") {
    val srcPtx = layout.projectDirectory.dir("../bin").file("kernels.ptx")
    val dest = layout.buildDirectory.dir("resources/main/bin")
    doFirst {
        if (srcPtx.asFile.exists()) {
            from(srcPtx)
            into(dest)
        }
    }
}

tasks.named("processResources") {
    dependsOn("ensureKernelsPtx")
}

// Task to download COCO dataset first
tasks.register<JavaExec>("downloadCOCO") {
    group = "application"
    description = "Download COCO validation dataset for YOLO training"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.DownloadCOCODataset")
}

// Task to train YOLO on COCO
tasks.register<JavaExec>("trainYOLO") {
    group = "application"
    description = "Train YOLO on COCO dataset (auto-downloads if needed)"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.TrainYOLOCoco")
}

// Task to train ALL object detection models on COCO
tasks.register<JavaExec>("trainAllDetectors") {
    group = "application"
    description = "Train YOLO + SSD + RetinaNet + Faster R-CNN on COCO"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.TrainAllDetectorsCoco")
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "8g"
}

tasks.register<JavaExec>("benchmarkResNet") {
    group = "benchmark"
    description = "Run benchmark for ResNet18 on CIFAR-10"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.BenchmarkResNetCifar10")
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "6g"
}

tasks.register<JavaExec>("benchmarkSentiment") {
    group = "benchmark"
    description = "Run benchmark for LSTM sentiment on RT-Polarity"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.BenchmarkSentiment")
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "6g"
}

tasks.register<JavaExec>("benchmarkDl4jResNet") {
    group = "benchmark"
    description = "Run DL4J benchmark for ResNet18 on CIFAR-10"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.BenchmarkDl4jResNetCifar10")
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "6g"
}

tasks.register<JavaExec>("benchmarkDl4jSentiment") {
    group = "benchmark"
    description = "Run DL4J benchmark for LSTM sentiment on RT-Polarity"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.BenchmarkDl4jSentiment")
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "6g"
}

tasks.register<JavaExec>("benchmarkMemoryPool") {
    group = "benchmark"
    description = "Run MemoryPool auto-expand benchmark"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.BenchmarkMemoryPool")
    jvmArgs("--add-modules=jdk.incubator.vector")
    maxHeapSize = "6g"
}

// Task to run UIT-VSFC example
tasks.register<JavaExec>("exampleUitVsfc") {
    group = "application"
    description = "Run UIT-VSFC Vietnamese sentiment & topic classification example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.user.nn.examples.TrainUitVsfc")
    jvmArgs("--add-modules=jdk.incubator.vector")
    System.getProperty("forceEpochs")?.let { systemProperty("forceEpochs", it) }
    maxHeapSize = "6g"
}

// Register tasks for all models in run_all.ps1
val trainTasks = mapOf(
    "runTrainIris" to "TrainIris",
    "runTrainLeNet" to "TrainLeNet",
    "runTrainFashionMNIST" to "TrainFashionMNIST",
    "runTrainCifar10" to "TrainCifar10",
    "runTrainResNet" to "TrainResNetCifar10",
    "runTrainViTCifar10" to "TrainViTCifar10",
    "runTrainSentiment" to "TrainSentiment",
    "runTrainUitVsfcMultitask" to "TrainUitVsfcMultitask",
    "runDashboardE2E" to "DashboardE2ETest"
)

trainTasks.forEach { (taskName, className) ->
    tasks.register<JavaExec>(taskName) {
        group = "application"
        description = "Run $className example"
        classpath = sourceSets["main"].runtimeClasspath
        mainClass.set("com.user.nn.examples.$className")
        jvmArgs("--add-modules=jdk.incubator.vector")
        System.getProperty("forceEpochs")?.let { systemProperty("forceEpochs", it) }
        maxHeapSize = "6g"
    }
}

