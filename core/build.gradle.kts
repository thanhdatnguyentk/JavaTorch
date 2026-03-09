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
    useJUnitPlatform()
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

tasks.named("processResources") {
    dependsOn("ensureKernelsPtx")
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
