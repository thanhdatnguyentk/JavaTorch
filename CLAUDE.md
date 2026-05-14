# CLAUDE.md — JavaTorch Development Guide

## 🧠 Project Overview

**JavaTorch** is a from-scratch Java Deep Learning framework mimicking PyTorch's API design.
It supports CPU (with SIMD via `jdk.incubator.vector`) and GPU (via JCuda/cuDNN) computation,
autograd-based backpropagation, and end-to-end training pipelines for CV and NLP tasks.

- **Language:** Java 21 (Toolchain enforced)
- **Build System:** Gradle 8.10+ (Kotlin DSL, multi-module)
- **GPU Backend:** JCuda 12.0.0 + cuDNN (NVIDIA only)
- **Package Root:** `com.user.nn`

---

## 📁 Project Structure

```
ML_framework/
├── build.gradle.kts          # Root build config (multi-module)
├── settings.gradle.kts       # include("core", "examples", "tests")
├── gradle.properties         # javaVersion=21, jcudaVersion=12.0.0
│
├── src/com/user/nn/          # ALL source code (consumed by :core)
│   ├── core/                 # Foundation: Tensor, Torch, Module, NN, Functional, CUDAOps
│   │   ├── Tensor.java       # Core tensor class (AutoCloseable, GPU sync, autograd)
│   │   ├── Torch.java        # Static ops: matmul, relu, softmax, conv2d, etc. (~3300 lines)
│   │   ├── Module.java       # Base class for all neural network modules
│   │   ├── Parameter.java    # Wraps a Tensor as a learnable parameter
│   │   ├── Functional.java   # Functional API (stateless ops)
│   │   ├── CUDAOps.java      # JCuda/cuDNN GPU kernel bindings
│   │   ├── GpuMemoryPool.java # Arena-style GPU memory allocator
│   │   ├── MemoryScope.java  # Scoped GPU memory management (try-with-resources)
│   │   └── kernels.cu        # Custom CUDA PTX kernels
│   │
│   ├── layers/               # Linear, Conv2d, ConvTranspose2d, Embedding, Dropout, Flatten, etc.
│   ├── activations/          # ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, ELU, LeakyReLU, etc.
│   ├── containers/           # Sequential, ModuleList, ModuleDict
│   ├── norm/                 # BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm
│   ├── pooling/              # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, ZeroPad2d
│   ├── rnn/                  # LSTM, GRU, RNN (+ Cell variants)
│   ├── attention/            # MultiheadAttention, TransformerEncoderLayer
│   ├── losses/               # CrossEntropyLoss, BCELoss, FocalLoss, L1Loss, KLDivLoss, etc.
│   ├── optim/                # SGD (with momentum), Adam optimizers + LR Schedulers
│   ├── metrics/              # Accuracy, MSE, MAE, MetricTracker, Evaluator
│   ├── models/
│   │   ├── cv/               # LeNet, VGG, ResNet, ViT, YOLO, SSD, RetinaNet, FasterRCNN
│   │   └── generative/       # GAN, VAE
│   ├── dataloaders/          # MNIST, CIFAR-10, UIT-VSFC, MovieComment, AnimeFace
│   ├── predict/              # Predictor, ImagePredictor, TextPredictor, BatchPredictor
│   └── utils/                # Dashboard, Progress, Visualization, COCODatasetDownloader
│
├── tests/                    # :tests module
│   ├── build.gradle.kts      # JUnit 5 config, GPU tag exclusion
│   └── java/com/user/nn/    # 49 test files, 100% JUnit 5
│       ├── TestTensor.java, TestContainers.java, TestAutogradConv.java, ...
│       └── (GPU tests tagged with @Tag("gpu"))
│
├── examples/                 # :examples module (training scripts)
│   ├── build.gradle.kts      # JavaExec tasks for training
│   └── src/                  # TrainFashionMNIST, TrainCifar10, TrainSentiment, TrainUitVsfc, etc.
│
├── core/                     # :core module (library build + publishing)
│   └── build.gradle.kts      # Dependencies, PTX compilation, Maven publishing
│
└── bin/                      # Compiled CUDA PTX kernels
```

---

## ⚙️ Build & Run Commands

### Prerequisites
- **Java 21** (auto-downloaded by Gradle toolchain)
- **NVIDIA GPU + CUDA 12.0** (optional, for GPU tests/training)
- **nvcc** on PATH (optional, for PTX kernel compilation)

### Core Commands

```bash
# Compile everything
./gradlew build

# Run CPU-only unit tests (default — GPU tests excluded)
./gradlew :tests:test

# Run ALL tests including GPU
./gradlew :tests:test -PincludeGPU=true

# Clean and re-run tests
./gradlew :tests:cleanTest :tests:test

# Run a specific test class
./gradlew :tests:test --tests "com.user.nn.TestTensor"

# Run training examples
./gradlew :examples:trainFashionMNIST
./gradlew :examples:trainCifar10
./gradlew :examples:trainSentiment
./gradlew :examples:exampleUitVsfc

# Run benchmarks
./gradlew :examples:benchmarkResNet
```

> **Important:** All Java commands MUST include `--add-modules=jdk.incubator.vector` JVM arg.
> This is already configured in `build.gradle.kts` for all tasks.

---

## 🏛️ Architecture & Key Patterns

### 1. Tensor System (`core/Tensor.java`)
- **Row-major float[] storage** with shape tracking
- **Dual-device:** CPU (`float[] data`) ↔ GPU (`Pointer deviceData`)
- **Autograd:** `requires_grad`, `grad`, `grad_fn` (reverse-mode AD via topological sort)
- **In-place safety:** Version counter (`_version`) detects mutations during backward
- **Memory management:** `AutoCloseable` + `Cleaner` safety net for GPU memory
- **GPU Memory Pool:** `MemoryScope` + `GpuMemoryPool` for arena-style GPU allocation

```java
// Pattern: Create tensor, compute, backward
Tensor x = Torch.randn(new int[]{2, 3});
x.requires_grad = true;
Tensor y = Torch.matmul(x, Torch.randn(new int[]{3, 1}));
Tensor loss = Torch.sum_tensor(y);
loss.backward();
// x.grad now contains gradients
```

### 2. Module System (`core/Module.java`)
- **Inheritance-based:** All layers extend `Module`
- **Parameter registration:** `addParameter(name, param)` / `addModule(name, module)`
- **Serialization:** Binary format via `save(path)` / `load(path)`
- **Device transfer:** `module.toGPU()` / `module.toCPU()` recursively moves all parameters
- **Train/Eval modes:** `module.train()` / `module.eval()` (affects Dropout, BatchNorm)

```java
// Pattern: Build a model
Sequential model = new Sequential(
    new Conv2d(1, 32, 3, 1, 1, true),
    new ReLU(),
    new MaxPool2d(2, 2, 0),
    new Flatten(),
    new Linear(32 * 14 * 14, 10, true)
);
model.toGPU(); // Move to GPU
```

### 3. Torch Static API (`core/Torch.java`)
The central ops hub (~3300 lines). Key categories:

| Category | Methods |
|----------|---------|
| **Creation** | `zeros`, `ones`, `rand`, `randn`, `full`, `arange`, `eye`, `tensor` |
| **Math** | `add`, `sub`, `mul`, `div`, `matmul`, `pow`, `exp`, `log`, `sqrt` |
| **Activations** | `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softmax`, `log_softmax` |
| **Reductions** | `sum`, `mean`, `max`, `min`, `argmax`, `argmin` |
| **Shape** | `permute`, `stack`, `cat`, `split`, `chunk`, `gather`, `scatter` |
| **Pooling** | `max_pool1d`, `avg_pool1d`, `adaptive_avg_pool2d`, `max_pool2d` |
| **Init** | `Torch.nn.init.kaiming_uniform_()`, `xavier_uniform_()`, etc. |
| **Grad** | `no_grad()`, `enable_grad()`, `is_grad_enabled()` |

### 4. GPU Code Path
- **Auto-detection:** `Torch.hasGPU()` / `CUDAOps.isAvailable()`
- **Transparent dispatch:** Most ops check `tensor.isGPU()` and route to CUDAOps
- **cuDNN integration:** Conv2d, Pooling use cuDNN via `CUDAOps.conv2dForward()`, etc.
- **Custom PTX kernels:** Element-wise ops compiled from `kernels.cu`
- **Stream management:** Compute stream + cuBLAS/cuDNN handles bound together

### 5. Functional API (`core/Functional.java`)
Stateless wrappers delegating to `Torch.*`. Supports both `Tensor` and legacy `NN.Mat` inputs.

---

## 🧪 Testing Standards

### Test Location & Structure
- **All tests:** `tests/java/com/user/nn/Test*.java`
- **Framework:** JUnit Jupiter 5.10.0
- **49 test files**, all structured as proper `@Test` methods (zero `main()` runners)

### Writing New Tests

```java
package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;

public class TestNewFeature {

    @Test
    void testBasicOperation() {
        Tensor a = Torch.tensor(new float[]{1f, 2f, 3f}, 3);
        Tensor b = Torch.tensor(new float[]{4f, 5f, 6f}, 3);
        Tensor c = Torch.add(a, b);
        assertEquals(5f, c.data[0], 1e-6, "Element 0 should be 5.0");
    }

    @Tag("gpu")
    @Test
    void testGPUOperation() {
        if (!Torch.hasGPU()) return; // Guard
        Tensor a = Torch.randn(new int[]{2, 3}).toGPU();
        Tensor b = Torch.randn(new int[]{2, 3}).toGPU();
        Tensor c = Torch.add(a, b);
        c.toCPU();
        assertEquals(6, c.numel());
    }
}
```

### Testing Rules

| Rule | Description |
|------|-------------|
| **Float delta** | Always use `assertEquals(expected, actual, 1e-5)` for float comparisons |
| **No `main()`** | All tests must be `@Test` methods |
| **No `System.exit`** | Let assertions fail naturally |
| **No `System.out` validation** | Use assertion messages instead |
| **GPU Guard** | Tag GPU tests with `@Tag("gpu")` |
| **Independence** | No shared mutable state between tests |
| **Determinism** | Use `Torch.manual_seed(42)` when testing random operations |
| **Shape assertions** | Always verify output shapes: `assertArrayEquals(expectedShape, result.shape)` |

### Test Tags

| Tag | Purpose | Default |
|-----|---------|---------|
| `gpu` | Requires NVIDIA GPU | **Excluded** |
| `gpu-smoke` | Fast GPU sanity checks | Excluded |
| `gpu-nightly` | Full GPU regression | Excluded |
| `slow` | Tests taking > 5 seconds | Included |
| `integration` | End-to-end flow tests | Included |

---

## 🔧 Coding Conventions

### General
- **Package:** All code under `com.user.nn.*`
- **Encoding:** UTF-8 everywhere
- **No deprecated APIs:** Avoid `finalize()`, use `Cleaner`
- **AutoCloseable:** GPU tensors implement `close()` — use try-with-resources when appropriate

### Tensor Operations
```java
// ✅ Correct: Use Torch static methods
Tensor result = Torch.matmul(a, b);
Tensor activated = Torch.relu(x);

// ✅ Correct: In-place ops end with underscore
tensor.add_(2.0f);
tensor.mul_(scalar);

// ✅ Correct: Autograd pattern
x.requires_grad = true;
Tensor loss = computeLoss(x);
loss.backward();
// Access x.grad

// ❌ Wrong: Direct data manipulation without marking dirty
tensor.data[0] = 5.0f; // WRONG — use tensor.set(5.0f, 0)

// ❌ Wrong: Forgetting GPU sync
float val = gpuTensor.data[0]; // WRONG — call toCPU() first
```

### Module Implementation Pattern
```java
public class MyLayer extends Module {
    private Parameter weight;
    
    public MyLayer(int inSize, int outSize) {
        Tensor w = new Tensor(inSize, outSize);
        Torch.nn.init.kaiming_uniform_(w);
        this.weight = new Parameter(w);
        addParameter("weight", this.weight);
    }
    
    @Override
    public Tensor forward(Tensor x) {
        return Torch.matmul(x, weight.getTensor());
    }
}
```

### GPU Memory Management
```java
// ✅ Pattern: Use MemoryScope for training loops
try (MemoryScope scope = new MemoryScope()) {
    Tensor input = batchData.toGPU();
    Tensor output = model.forward(input);
    Tensor loss = criterion.forward(output, target);
    loss.backward();
    optimizer.step();
} // GPU memory auto-released

// ✅ Pattern: Model parameters allocated OUTSIDE MemoryScope
// so they persist across iterations
model.toGPU(); // Parameters use cudaMalloc, not pool
```

---

## 🚨 Known Gotchas & Anti-Patterns

### Critical
1. **GPU ↔ CPU sync:** Always call `.toCPU()` before reading `.data[]` from a GPU tensor
2. **MemoryScope placement:** Model parameters MUST be allocated outside `MemoryScope`
   (otherwise pool reset overwrites weights)
3. **Conv2d input shape:** Expects `[N, C, H, W]` 4D tensors; 2D inputs auto-reshape only
   if `H*W*C == flat_size`
4. **Autograd no_grad:** Wrap inference in `Torch.no_grad()` / `Torch.enable_grad()` to
   avoid building computation graphs during evaluation

### Common Mistakes
- **Silent NaN:** Check for division by zero in loss functions (use epsilon `1e-12`)
- **Shape mismatch in matmul:** `Torch.matmul(a, b)` requires `a.shape[-1] == b.shape[-2]`
- **Forgetting `model.eval()`:** Dropout and BatchNorm behave differently in eval mode
- **Double-free GPU memory:** Don't `close()` pool-managed tensors manually — handled by `MemoryScope`

---

## 📦 Dependencies (managed in `core/build.gradle.kts`)

| Dependency | Version | Purpose |
|------------|---------|---------|
| JCuda | 12.0.0 | CUDA runtime bindings |
| JCublas | 12.0.0 | cuBLAS matrix operations |
| JCudnn | 12.0.0 | cuDNN neural network primitives |
| JavaCPP | 1.5.10 | OpenBLAS native bridge |
| OpenBLAS | 0.3.26-1.5.10 | CPU BLAS operations |
| DL4J | 0.9.1 | Baseline comparison (Phase 2) |
| JFreeChart | 1.5.4 | Training visualization |
| Javalin | 5.6.3 | Web dashboard server |
| Jackson | 2.15.2 | JSON serialization |
| JUnit Jupiter | 5.10.0 | Testing framework |

---

## 🔄 Development Workflow

### Adding a New Layer
1. Create `src/com/user/nn/layers/MyNewLayer.java` extending `Module`
2. Register parameters in constructor via `addParameter()`
3. Implement `forward(Tensor x)` with autograd support
4. Create `tests/java/com/user/nn/TestMyNewLayer.java`
5. Run tests: `./gradlew :tests:test --tests "com.user.nn.TestMyNewLayer"`
6. Run full suite: `./gradlew :tests:test` to ensure no regressions

### Adding a New Loss Function
1. Create `src/com/user/nn/losses/MyLoss.java` extending `Module`
2. Delegate to `Functional` or implement in `Torch` / `Functional` classes
3. Include autograd `grad_fn` in the `Tensor` returned
4. Write tests verifying both forward value and gradient correctness
5. Run tests

### Fixing a Bug
1. Identify the failing test or behavior
2. Write a **regression test first** that reproduces the bug
3. Fix the code
4. Run: `./gradlew :tests:cleanTest :tests:test`
5. Verify ALL tests pass

---

## 🏷️ Quick Reference: Import Patterns

```java
// Core
import com.user.nn.core.*;

// Layers
import com.user.nn.layers.*;

// Activations
import com.user.nn.activations.*;

// Containers
import com.user.nn.containers.*;

// Losses
import com.user.nn.losses.*;

// Optimizers
import com.user.nn.optim.Optim;

// Normalization
import com.user.nn.norm.*;

// Pooling
import com.user.nn.pooling.*;

// RNN
import com.user.nn.rnn.*;

// Testing
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
```

---

## 📊 Current Status (May 2026)

- ✅ **50 test files**, all JUnit 5 — zero `main()` test runners
- ✅ **118+ test methods**, CPU test pass rate 100% via `./gradlew :tests:test`
- ✅ **GPU stable** — Autograd verified across Conv2d, LSTM, ViT, etc.
- ✅ **End-to-end training:** FashionMNIST, CIFAR-10, Sentiment, UIT-VSFC (single & multi-task)
- ✅ **Web Dashboard** for real-time training visualization
- ✅ **Object Detection:** YOLO, SSD, RetinaNet, Faster R-CNN architectures
- ✅ **Generative:** GAN (MNIST, Anime), VAE
- ✅ **NLP Multi-task:** LSTM vs Transformer comparison on UIT-VSFC
