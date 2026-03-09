plugins {
    `java`
    `maven-publish`
}

allprojects {
    repositories {
        mavenCentral()
    }
}

subprojects {
    apply(plugin = "java")

    group = "com.user.nn"
    version = property("projectVersion") as String

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of((property("javaVersion") as String).toInt()))
        }
    }

    tasks.withType(JavaCompile::class.java) {
        options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
    }

    tasks.register("sourcesJar", Jar::class) {
        archiveClassifier.set("sources")
        from(sourceSets.main.get().allSource)
    }

    tasks.register("javadocJar", Jar::class) {
        archiveClassifier.set("javadoc")
        from(tasks.named("javadoc"))
    }
}
