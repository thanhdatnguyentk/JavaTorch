# ML_framework — Java Neural Network Library (PyTorch-inspired)

A minimal re-implementation of core PyTorch concepts in pure Java for learning and experimentation.

## Key Features

| Feature | Status |
|---------|--------|
| **Tensor API** | ✅ Full shape operations, broadcasting, indexing, math, reductions |
| **Autograd Engine** | ✅ `requires_grad`, `backward()`, `grad_fn` chain |
| **nn.Module System** | ✅ `Module`, `Parameter`, `Sequential`, `ModuleList`, `ModuleDict` |
| **Linear Layers** | ✅ `Linear` with Tensor autograd |
| **Activations** | ✅ `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softplus`, `Dropout` |
| **CNN Layers** | ✅ `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `AvgPool2d`, `ZeroPad2d` — all with autograd |
| **Normalization** | ✅ `BatchNorm1d`, `LayerNorm`, `InstanceNorm` |
| **Optimizers** | ✅ `optim.SGD` (momentum), `optim.Adam` |
| **Loss Functions** | ✅ `cross_entropy_tensor`, `nll_loss`, `mse_loss_tensor`, `huber_loss` |
| **RNN/LSTM** | ✅ `RNNCell`, `LSTMCell`, `RNN`, `LSTM` (with BPTT) |
| **Data Loader** | ✅ `Dataset`, `DataLoader` (Multi-threaded Producer-Consumer) |
| **SIMD Support** | ✅ Java Vector API (AVX2/AVX-512) for `matmul`, `dot` |
| **GPU/cuDNN** | ✅ **JCuda + JCudnn** integration for high-performance CNN training with **GPU-accelerated Backward Pass** (cuDNN), **GPU Compatibility Audit** complete |
| **NLP Support** | ✅ `Embedding`, `Vocabulary`, `BasicTokenizer`, `SentimentModel` |
| **Autograd Optimized** | ✅ **Topological Sort** for $O(N)$ backprop (No more recursive $O(2^N)$ hangs) |
| **GPU Optimization** | ✅ **Kernel Fusion**, **CUDA Streams**, **Arena Memory Pool**, **Custom PTX** |
| **Test Suite** | ✅ 40 automated tests fully operational |

## Quick Start

```bash
# Compile core framework (Note: --add-modules required for SIMD Vector API)
javac --add-modules jdk.incubator.vector -d bin src/com/user/nn/core/*.java src/com/user/nn/optim/*.java src/com/user/nn/dataloaders/*.java src/com/user/nn/models/*.java

# Run all tests (36 tests)
powershell -File tests/run-tests.ps1

# Examples:
# Compile & run Iris example
javac --add-modules jdk.incubator.vector -d bin -cp bin src/com/user/nn/examples/TrainIris.java
java --add-modules jdk.incubator.vector -cp bin com.user.nn.examples.TrainIris

# Compile & run Fashion-MNIST CNN (Using Multi-threaded DataLoader)
javac --add-modules jdk.incubator.vector -d bin -cp bin src/com/user/nn/examples/TrainFashionMNIST.java
java --add-modules jdk.incubator.vector -cp bin com.user.nn.examples.TrainFashionMNIST

# Compile & run CIFAR-10 CNN (Using Multi-threaded DataLoader)
javac --add-modules jdk.incubator.vector -d bin -cp bin;lib/* src/com/user/nn/examples/TrainCifar10.java
java --add-modules jdk.incubator.vector -cp bin;lib/* com.user.nn.examples.TrainCifar10

# Compile & run ResNet-18 on CIFAR-10 (High performance GPU training)
javac --add-modules jdk.incubator.vector -d bin -cp bin;lib/* src/com/user/nn/examples/TrainResNetCifar10.java
java --add-modules jdk.incubator.vector -cp bin;lib/* com.user.nn.examples.TrainResNetCifar10

# Compile & run Sentiment Analysis (Real Movie Comments Dataset)
javac --add-modules jdk.incubator.vector -d bin -cp bin src/com/user/nn/examples/TrainSentiment.java
java --add-modules jdk.incubator.vector -cp bin com.user.nn.examples.TrainSentiment
```

## Project Structure

```
src/
└── com/user/nn/
    ├── core/
    │   ├── NN.java          # Module, Parameter, layers, F (loss functions), RNN/LSTM, **Embedding**, **Dropout**
    │   ├── Tensor.java      # Core Tensor class with **Topological Sort Autograd**
    │   └── Torch.java       # Tensor utilities (creation, ops, broadcasting, SIMD)
    ├── optim/
    │   └── Optim.java       # Optimizers (SGD, Adam)
    ├── metrics/
    │   ├── Metric.java      # Metric interface
    │   ├── Accuracy.java    # Classification accuracy
    │   ├── MeanSquaredError.java # Regression MSE
    │   └── MetricTracker.java # Epoch state tracking
    ├── dataloaders/
    │   ├── Data.java        # Dataset, DataLoader, **Vocabulary**, **BasicTokenizer**
    │   ├── MnistLoader.java # Download/Parse Fashion-MNIST
    │   ├── Cifar10Loader.java # Download/Parse CIFAR-10
    │   └── MovieCommentLoader.java # Download/Parse Movie Polarity Dataset
    ├── models/
    │   └── SentimentModel.java  # Embedding -> LSTM -> Dropout -> Linear Model
    └── examples/
        ├── App.java
        ├── TrainSentiment.java  # Sentiment Analysis Training on Real Data
        ├── TrainIris.java       # Simple MLP for Iris classification
        ├── TrainFashionMNIST.java # MLP with DataLoader for Fashion-MNIST
        ├── TrainCifar10.java    # CNN with DataLoader on CIFAR-10
        └── TestVectorBenchmark.java # SIMD vs Scalar Matmul Benchmark
tests/
├── run-tests.ps1        # Automated test runner (36 tests)
├── java/com/user/nn/    # Unit test files (including TestRNN, TestDropout, TestBatch*)
└── *.py                 # NumPy reference scripts
```

## Roadmap

### ✅ Phase 1–8: Core & Ecosystem (Complete)
- Tensor API, Autograd engine, NN.Module migration
- CNN Autograd Migration (`Conv2d`, `MaxPool2d`, `AvgPool2d`, `ZeroPad2d`, `ConvTranspose2d`)
- `optim.SGD`, `optim.Adam`
- Loss Functions (`nll_loss`, `mse_loss_tensor`, `huber_loss`)
- Normalization (`LayerNorm`, `InstanceNorm`)

### ✅ Phase 9: Sequential Models (Complete)
- **RNN & LSTM**: ✅ `RNNCell`, `LSTMCell`, `RNN`, `LSTM` implemented with BPTT support.
- **Transformer**: 🔲 MultiheadAttention, Softmax-dim, EncoderLayer (Next Step).

### ✅ Phase 10: System Optimizations (Complete)
- **DataLoader**: ✅ Multi-threaded Producer-Consumer pipeline (`data.java`).
- **SIMD**: ✅ Java Vector API integration for hardware-accelerated `matmul` and `dot`.
- Added `TestVectorBenchmark` showing 1024x1024 Matmul in ~112ms.

### ✅ Phase 11: NLP & Autograd Optimization (Complete)
- **Embedding**: Added `Embedding` layer with full backprop.
- **NLP Utilities**: `Vocabulary` and `BasicTokenizer` added to `data.java`.
- **Topological Sort**: Rewrote `Tensor.backward()` to use non-recursive topological sorting, improving RNN/LSTM backprop complexity from $O(2^N)$ to $O(N)$.
- **Movie Dataset**: Added `MovieCommentLoader` for Rotten Tomatoes dataset.
- **Sentiment Analysis**: End-to-end training achieved with `TrainSentiment.java`.

### ✅ Polish & Real Datasets (Complete)
- Added `TrainFashionMNIST.java` (89.04% Test Accuracy with GPU acceleration)
- Added `TrainCifar10.java` (Dramatic speed-up via **cuDNN**: ~10s/epoch vs several minutes)
- **Train/Eval Modes**: Integrated `model.train()` and `model.eval()` to support inference-specific behaviors like Dropout and BatchNorm.
- **Dropout**: Implemented `NN.Dropout` and `Torch.dropout` with inverted dropout scaling.

### ✅ Phase 12: Metrics Tracking (Complete)
- **metrics**: ✅ Standardized `Accuracy`, `MeanSquaredError`, and `MeanAbsoluteError`.
- **MetricTracker**: ✅ Refactored all training examples to use decoupled metric tracking for cleaner code.

### ✅ Phase 13: GPU & cuDNN Acceleration (Complete)
- **JCuda Core**: ✅ Integrated JCuda (12.0.0) for GPU-aware tensors.
- **Memory Management**: ✅ Implement `AutoCloseable` tensors with `cudaFree` for GPU memory leak prevention.
- **CUBLAS Integration**: ✅ GPU-accelerated `matmul` using `cublasSgemm`.
- **JCudnn Support**: ✅ **Conv2d**, **MaxPool2d**, and **ReLU** accelerated via cuDNN (8.9.x).
- **Hybrid Dispatch**: ✅ Device-aware `Torch.java` / `NN.java` routes math to GPU if tensors reside there.
- **Backward Sync**: ✅ Automatic CPU synchronization for backward pass when GPU forward is used.
- **Full Compatibility Audit**: ✅ Comprehensive audit of all math and NN operations to ensure device-aware dispatch and automatic synchronization.
- **Convenience API**: ✅ Added `Tensor.to(Device)` for seamless device migration.
- Full suite of 40 tests operational (including cuDNN initialization and GPU forward verification).

### ✅ Phase 14: Advanced GPU Optimizations (Zero-Overhead) (Complete)
- **Kernel Fusion**: ✅ Integrated `Conv2d + Bias + ReLU` fusion using `cudnnConvolutionBiasActivationForward`.
- **CUDA Streams**: ✅ Implemented asynchronous Compute and Transfer streams to overlap I/O and computation.
- **Arena Memory Pool**: ✅ Developed `GpuMemoryPool` for instant raw VRAM allocation (0ms overhead).
- **Auto-Scaling**: ✅ Pool size automatically scales based on model parameters or available VRAM.
- **Custom PTX Kernels**: ✅ Implemented element-wise operations (Add, Sub, Mul) as native GPU kernels to eliminate CPU Fallback.
- **MemoryScope**: ✅ Automated tensor lifecycle management for zero-leak training loops.
- **GPU Backward**: ✅ Implemented full cuDNN backward support for `Conv2d` (BackwardData, BackwardFilter, BackwardBias), eliminating CPU synchronization in CNN training.

### ✅ Phase 15: Computer Vision & ResNet (Complete)
- **VGG & ResNet**: ✅ Configurable VGG/ResNet architectures with skip-connections and autograd.
- **ResNet-18 Performance**: ✅ Achieved ~66% Accuracy on CIFAR-10 in 2 epochs (~15 mins) using end-to-end GPU training.
- **Evaluator Class**: ✅ Centralized model evaluation with multi-threaded data fetching.
- **Global Average Pooling**: ✅ `adaptive_avg_pool2d` supports flexible architectural endpoints.
- **Transformer**: MultiheadAttention, Softmax-dim, EncoderLayer.
- Conv1d/Conv3d, BatchNorm2d/3d, GroupNorm
- Learning rate schedulers
- JUnit integration

## Test Suite (37 tests)

| Test | Coverage |
|------|----------|
| TestGPUMatmul | JCuda/GPU Data transfer & Matmul correctness |
| TestMatOps | Matrix operations |
| TestContainers | Sequential, ModuleList, ModuleDict |
| TestParameterAndModules | Parameter, Module base |
| TestFunctional | F utility functions |
| TestLinearReLU | Linear+ReLU vs PyTorch reference |
| TestActivations | All activation layers |
| TestLossesAndNorms | Legacy losses & BatchNorm |
| TestConvPool | Conv2d + MaxPool2d vs Python ref |
| TestTorchCoverage | Torch utility functions |
| TestTorchExtras | Advanced Torch ops |
| TestTensor | Tensor basics |
| TestGatherScatterExtras | Gather/Scatter ops |
| TestAutogradSimple | Basic autograd |
| TestAutogradLinear | Linear layer autograd |
| TestAutogradShapeOps | Shape operation gradients |
| TestAutogradReductions | Reduction gradients |
| TestAutogradMatmul | Matrix multiply gradients |
| TestAutogradActivations | Activation gradients |
| TestAutogradMLP | End-to-end MLP training |
| TestAutogradConv | Conv2d, MaxPool2d, AvgPool2d, ZeroPad2d, ConvTranspose2d gradients |
| TestOptimizers | SGD + Adam convergence |
| TestLossFunctions | NLL, MSE, Huber forward+backward |
| TestNormLayers | LayerNorm, InstanceNorm forward+backward |
| TestMetrics | Metrics calculations |
| TestRNN | RNN/LSTM Forward + Backpropagation Through Time |
| TestAutogradEmbedding | Embedding layer autograd |
| TestDropout | Training/Eval configurations and Dropout |
| TestBatch1 | Combined batch 1 tests |
| TestBatch2 | Combined batch 2 tests |
| TestBatch3 | Combined batch 3 tests |
| TestBatch4 | Combined batch 4 tests |

---

*Last updated: 2026-03-08*
