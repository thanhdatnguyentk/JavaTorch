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
