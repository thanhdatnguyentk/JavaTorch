plugins {
    `java-library`
    `maven-publish`
}

val jcudaClassifier = run {
    val os = org.gradle.internal.os.OperatingSystem.current()
    val osPart = when {
        os.isWindows -> "windows"
        os.isLinux -> "linux"
        os.isMacOsX -> "apple"
        else -> throw GradleException("Unsupported OS for JCuda natives: ${System.getProperty("os.name")}")
    }
    val archPart = when {
        System.getProperty("os.arch").contains("64") -> "x86_64"
        else -> "x86"
    }
    "$osPart-$archPart"
}

configurations.configureEach {
    exclude(group = "org.jcuda", module = "jcuda-natives")
    exclude(group = "org.jcuda", module = "jcublas-natives")
    exclude(group = "org.jcuda", module = "jcudnn-natives")
}

dependencies {
    implementation("org.bytedeco:javacpp:${property("javacppVersion")}")
    implementation("org.bytedeco:openblas:${property("openblasVersion")}")
    runtimeOnly("org.bytedeco:openblas:${property("openblasVersion")}:windows-x86_64")
    implementation("org.jcuda:jcuda:${property("jcudaVersion")}") {
        exclude(group = "org.jcuda", module = "jcuda-natives")
    }
    implementation("org.jcuda:jcublas:${property("jcudaVersion")}") {
        exclude(group = "org.jcuda", module = "jcublas-natives")
    }
    implementation("org.jcuda:jcudnn:${property("jcudaVersion")}") {
        exclude(group = "org.jcuda", module = "jcudnn-natives")
    }
    runtimeOnly("org.jcuda:jcuda-natives:${property("jcudaVersion")}:$jcudaClassifier")
    runtimeOnly("org.jcuda:jcublas-natives:${property("jcudaVersion")}:$jcudaClassifier")
    runtimeOnly("org.jcuda:jcudnn-natives:${property("jcudaVersion")}:$jcudaClassifier")
    
    // Visualization dependencies
    implementation("org.jfree:jfreechart:1.5.4")
    implementation("org.jfree:jfreesvg:3.4")
    
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}

// Keep existing src/ layout to avoid moving files initially
sourceSets {
    main {
        java {
            srcDirs(file("../src"))
            exclude("TestDims.java")
        }
    }
    test {
        java.srcDirs(file("src/test/java"))
    }
}

tasks.test {
    useJUnitPlatform {
        excludeTags("gpu-smoke", "gpu-nightly", "gpu-manual")
    }
    jvmArgs("--add-modules=jdk.incubator.vector")
}

tasks.register<Test>("gpuSmoke") {
    description = "Runs fast GPU smoke tests (skip with reason if CUDA unavailable)"
    group = "verification"
    useJUnitPlatform {
        includeTags("gpu-smoke")
    }
    jvmArgs("--add-modules=jdk.incubator.vector")
}

tasks.register<Test>("gpuNightly") {
    description = "Runs full GPU nightly test suite"
    group = "verification"
    useJUnitPlatform {
        includeTags("gpu-nightly")
    }
    jvmArgs("--add-modules=jdk.incubator.vector")
}

tasks.register<Copy>("ensureKernelsPtx") {
    val srcPtx = layout.projectDirectory.dir("../bin").file("kernels.ptx")
    val dest = layout.buildDirectory.dir("resources/main")
    doFirst {
        if (srcPtx.asFile.exists()) {
            from(srcPtx)
            into(dest)
        }
    }
}

tasks.register<Copy>("ensureTestKernelsPtx") {
    val srcPtx = layout.projectDirectory.dir("../bin").file("kernels.ptx")
    val dest = layout.buildDirectory.dir("resources/test")
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

tasks.named("processTestResources") {
    dependsOn("ensureTestKernelsPtx")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            artifact(tasks.named("sourcesJar"))
            artifact(tasks.named("javadocJar"))
        }
    }
}
