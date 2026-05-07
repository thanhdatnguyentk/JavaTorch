plugins {
    `java`
}

sourceSets {
    test {
        java.srcDirs(file("../tests/java"))
    }
}

configurations.configureEach {
    exclude(group = "org.jcuda", module = "jcuda-natives")
    exclude(group = "org.jcuda", module = "jcublas-natives")
    exclude(group = "org.jcuda", module = "jcudnn-natives")
}

dependencies {
    testImplementation(project(":core"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
    testImplementation("org.jcuda:jcuda:${property("jcudaVersion")}") {
        exclude(group = "org.jcuda", module = "jcuda-natives")
    }
    testImplementation("org.jcuda:jcublas:${property("jcudaVersion")}") {
        exclude(group = "org.jcuda", module = "jcublas-natives")
    }
    testImplementation("org.jcuda:jcudnn:${property("jcudaVersion")}") {
        exclude(group = "org.jcuda", module = "jcudnn-natives")
    }
}

tasks.test {
    useJUnitPlatform {
        // Exclude GPU tests by default
        if (!project.hasProperty("includeGPU")) {
            excludeTags("gpu")
        }
    }
    workingDir = rootProject.projectDir
    jvmArgs("--add-modules=jdk.incubator.vector")
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}

tasks.register<JavaExec>("runStandaloneTest") {
    val testClass = project.findProperty("testClass") as String? ?: "com.user.nn.TestBatch1"
    classpath = sourceSets.test.get().runtimeClasspath
    mainClass.set(testClass)
    
    // Ensure we use the toolchain java launcher if possible
    val javaToolchains = project.extensions.getByType<JavaToolchainService>()
    javaLauncher.set(javaToolchains.launcherFor {
        languageVersion.set(JavaLanguageVersion.of(21))
    })
    
    jvmArgs("--add-modules=jdk.incubator.vector")
    standardOutput = System.out
    errorOutput = System.err
}
