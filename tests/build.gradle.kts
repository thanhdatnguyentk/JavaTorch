plugins {
    `java`
}

sourceSets {
    test {
        java.srcDirs(file("../tests/java"))
    }
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}

tasks.test {
    useJUnitPlatform()
}
