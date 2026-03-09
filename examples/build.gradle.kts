plugins {
    application
}

dependencies {
    implementation(project(":core"))
}

// No single mainClass; users can run specific examples via Gradle 'run' with --args or create custom tasks
