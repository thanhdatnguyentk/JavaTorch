# JavaTorch - Pytorch but in java

[Tieng Viet](README.vn.md) | [Tutorial](TUTORIAL.md) | [Tutorial VI](TUTORIAL.vn.md) | [API Reference](API_REFERENCE.md) | [API Reference VI](API_REFERENCE.vn.md)

![Java](https://img.shields.io/badge/Java-21+-orange)
![Build](https://img.shields.io/badge/Build-Gradle%20Multi--Module-blue)
![CUDA](https://img.shields.io/badge/GPU-JCuda%20%2B%20cuDNN-green)
![CPU](https://img.shields.io/badge/CPU-Vector%20API%20%2B%20OpenBLAS-purple)
![Tests](https://img.shields.io/badge/Tests-44%20registered-success)

This is a Java machine learning framework inspired by PyTorch. It is designed for three goals at once: learning how deep learning frameworks work internally, training models directly in Java, and progressively scaling from CPU execution to GPU execution through JCuda, cuBLAS, and cuDNN.

The repository already includes a tensor engine, autograd, a `Module/Parameter` system, data loaders, optimizers, CNNs, RNNs, Transformers, mixed precision, OpenBLAS integration, custom CUDA kernels, and a fully passing regression suite.

## Getting Started

If you only want the shortest path to a working setup, run these three commands:

```powershell
gradle wrapper
.\gradlew.bat :core:test
.\gradlew.bat :core:build
```

Then continue with:

- `TUTORIAL.md` for the step-by-step onboarding guide.
- `API_REFERENCE.md` for the package-level API map.

## System Overview

```mermaid
flowchart LR
    A[Data Loaders] --> B[Tensor / Torch]
    B --> C[Autograd Graph]
    C --> D[Module / Layers]
    D --> E[Optim / Scheduler]
    B --> F[CPU Path]
    B --> G[GPU Path]
    F --> H[Vector API]
    F --> I[OpenBLAS]
    G --> J[JCuda / cuBLAS]
    G --> K[cuDNN]
    G --> L[PTX Kernels]
```

## Highlights

- Tensor engine with reshape, broadcasting, indexing, reductions, transpose, gather/scatter, `matmul`, and `bmm`.
- Dynamic graph autograd with `requires_grad`, `grad_fn`, `backward()`, topological traversal, and version checking for in-place ops.
- PyTorch-like module system with `Sequential`, `ModuleList`, `ModuleDict`, and `Parameter`.
- Broad layer support: `Linear`, `Embedding`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, pooling, normalization, attention, and transformer encoder blocks.
- CPU acceleration through the Java Vector API and OpenBLAS via JavaCPP/bytedeco.
- GPU acceleration through JCuda, cuBLAS, cuDNN, memory pools, CUDA streams, and custom PTX kernels.
- End-to-end examples for Iris, Fashion-MNIST, CIFAR-10, Sentiment Analysis, ViT, GAN, and VAE.
- 44 registered test classes currently passing in the PowerShell test runner.

## Reference Benchmarks

The numbers below were collected from the current repository state using the built-in benchmark tests. They are representative measurements, not universal guarantees.

| Task | Backend | Size | Latest measured result |
|---|---|---|---|
| Large CPU matmul | OpenBLAS | `256 x 256` | `0.58 ms / matmul` |
| Vectorized CPU matmul | Java Vector API | benchmark suite | `19.10 ms / matmul` |
| Regression suite | PowerShell runner | 44 test classes | full pass |

## Companion Docs

- `TUTORIAL.md`: step-by-step onboarding in English
- `TUTORIAL.vn.md`: Vietnamese tutorial
- `API_REFERENCE.md`: package-level API map in English
- `API_REFERENCE.vn.md`: package-level API map in Vietnamese

## Build and Release Notes

- The default workflow is now Gradle-based (`:core`, `:examples`, `:tests`).
- Full wrapper generation is supported via `gradle wrapper`.
- Verified release command:

```powershell
.\gradlew.bat :core:clean :core:test :core:build --no-daemon
```

- Latest full verification (2026-03-10):

```powershell
.\gradlew.bat :core:test :tests:test :core:gpuSmoke :core:gpuNightly --continue
```

Result: `BUILD SUCCESSFUL`

- Legacy PowerShell runner remains available for compatibility:

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
```

---

Documentation updated for the current codebase state on 2026-03-10.
