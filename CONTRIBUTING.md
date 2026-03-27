# Contributing to JavaTorch

We love your input! We want to make contributing to JavaTorch as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Adding new CUDA kernels, layers, or optimization strategies

## Getting Started with Development

This is a Gradle-based Java project that leverages JCuda and OpenBLAS. 

1. Ensure you have Java 21+ installed.
2. If you are developing GPU features, ensure CUDA toolkit and cuDNN are installed.
3. Build the project and run tests:
   ```powershell
   ./gradlew core:build
   ./gradlew core:test
   ```

## We Develop with GitHub
We use GitHub to host code, track issues and feature requests, and accept pull requests.

### Pull Request Process
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests to the `core/src/test/java` directory.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (`./gradlew test`).
5. Issue that pull request!

## Report bugs using GitHub's issue tracker
We use GitHub issues to track public bugs. Report a bug by opening a new issue; it's that easy!

## Write bug reports with detail, background, and sample code
**Great Bug Reports** tend to have:

- A quick summary and/or background
- Expected behavior vs. actual behavior
- A minimal reproducible code snippet (e.g., using `Tensor` API)
- Environment details (OS, Java version, CUDA version, GPU model)
- Stack traces or error logs if any!