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
    useJUnitPlatform()
}
