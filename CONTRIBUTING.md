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
   .\gradlew.bat :core:build
   .\gradlew.bat :tests:test
   ```

## Development Workflow

### Adding a New Feature
1. Write the implementation in the appropriate package under `src/com/user/nn/`.
2. Write a corresponding JUnit 5 test in `tests/java/com/user/nn/`.
3. Run the specific test: `.\gradlew.bat :tests:test --tests "com.user.nn.TestYourFeature"`
4. Run the full suite to check for regressions: `.\gradlew.bat :tests:cleanTest :tests:test`

### Fixing a Bug
1. Write a regression test that reproduces the bug **first**.
2. Fix the code.
3. Run: `.\gradlew.bat :tests:cleanTest :tests:test`
4. Verify ALL tests pass.

### Testing Standards
- All tests must use JUnit 5 `@Test` annotations (no `main()` runners).
- Use `assertEquals(expected, actual, 1e-5)` for float comparisons.
- Use `assertArrayEquals(expectedShape, result.shape)` for shape checks.
- Tag GPU tests with `@Tag("gpu")`.
- Use `Torch.manual_seed(42)` for deterministic tests.

## We Develop with GitHub
We use GitHub to host code, track issues and feature requests, and accept pull requests.

### Pull Request Process
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests to `tests/java/com/user/nn/`.
3. If you've changed APIs, update the documentation (`API_REFERENCE.md`, `TUTORIAL.md`).
4. Ensure the test suite passes (`.\gradlew.bat :tests:test`).
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

## Coding Conventions

- Package: all code under `com.user.nn.*`
- No deprecated APIs: avoid `finalize()`, use `Cleaner` for GPU memory
- Use `Torch.nn.init.*` for weight initialization
- In-place ops end with underscore: `add_()`, `mul_()`
- GPU tensors implement `AutoCloseable` — use `MemoryScope` in training loops
- See `CLAUDE.md` for full coding standards