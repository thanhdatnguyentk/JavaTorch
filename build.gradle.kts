plugins {
    `java-base`
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

    configure<JavaPluginExtension> {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of((property("javaVersion") as String).toInt()))
        }
    }

    tasks.withType<JavaCompile>().configureEach {
        options.encoding = "UTF-8"
        options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
    }

    tasks.register("sourcesJar", Jar::class) {
        archiveClassifier.set("sources")
        // Use lazy evaluation
        doFirst {
            from(project.extensions.getByType<SourceSetContainer>().getByName("main").allSource)
        }
    }

    tasks.register("javadocJar", Jar::class) {
        archiveClassifier.set("javadoc")
        from(tasks.named("javadoc"))
    }
}
