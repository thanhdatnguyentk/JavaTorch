plugins {
    `java-library`
}

dependencies {
    implementation("org.bytedeco:javacpp:${property("javacppVersion")}")
    implementation("org.bytedeco:openblas:${property("openblasVersion")}")
    implementation("org.jcuda:jcuda:${property("jcudaVersion")}")
    implementation("org.jcuda:jcublas:${property("jcudaVersion")}")
    implementation("org.jcuda:jcudnn:${property("jcudaVersion")}")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}

// Keep existing src/ layout to avoid moving files initially
sourceSets {
    main {
        java.srcDirs(file("../src"))
        resources.srcDirs(file("../bin"))
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
