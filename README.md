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
| **Test Suite** | ✅ 24 automated tests via `run-tests.ps1` |

## Quick Start

```bash
# Compile (Note: --add-modules required for SIMD Vector API)
javac --add-modules jdk.incubator.vector -d bin src/com/user/nn/*.java

# Run all tests (24 tests)
powershell -File tests/run-tests.ps1

# Examples:
# Compile & run Iris example
javac --add-modules jdk.incubator.vector -d bin -cp bin src/TrainIris.java
java --add-modules jdk.incubator.vector -cp bin TrainIris

# Compile & run Fashion-MNIST CNN (Using Multi-threaded DataLoader)
javac --add-modules jdk.incubator.vector -d bin -cp bin src/TrainFashionMNIST.java
java --add-modules jdk.incubator.vector -cp bin TrainFashionMNIST

# Compile & run CIFAR-10 CNN (Using Multi-threaded DataLoader)
javac --add-modules jdk.incubator.vector -d bin -cp bin src/TrainCifar10.java
java --add-modules jdk.incubator.vector -cp bin TrainCifar10
```

## Project Structure

```
src/
├── com/user/nn/
│   ├── nn.java          # Module, Parameter, layers, F (loss functions), RNN/LSTM
│   ├── Tensor.java      # Core Tensor class with autograd
│   ├── Torch.java       # Tensor utilities (creation, ops, broadcasting, SIMD)
│   ├── data.java        # *NEW* Dataset & DataLoader (Multithreading)
│   ├── optim.java       # Optimizers (SGD, Adam)
│   ├── MnistLoader.java # Download/Parse Fashion-MNIST
│   └── Cifar10Loader.java # Download/Parse CIFAR-10
├── TrainIris.java       # Simple MLP for Iris classification
├── TrainFashionMNIST.java # MLP with DataLoader for Fashion-MNIST
├── TrainCifar10.java    # CNN with DataLoader on CIFAR-10
├── TestVectorBenchmark.java # *NEW* SIMD vs Scalar Matmul Benchmark
tests/
├── run-tests.ps1        # Automated test runner (24 tests)
├── java/com/user/nn/    # Unit test files (including TestRNN)
└── *.py                 # NumPy reference scripts
```

## Roadmap

### ✅ Phase 1–8: Core & Ecosystem (Complete)
- Tensor API, Autograd engine, nn.Module migration
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

### ✅ Polish & Real Datasets (Complete)
- Added `TrainFashionMNIST.java` (87% Test Accuracy in 5 epochs)
- Added `TrainCifar10.java` (45% Test Accuracy in 2 epochs, pure Java CNN)
- Full suite of 24 tests fully operational

### 🔲 Future Work
- Conv1d/Conv3d, BatchNorm2d/3d, GroupNorm
- Learning rate schedulers
- JUnit integration
- Data loading utilities (Dataset & DataLoader abstractions)

## Test Suite (23 tests)

| Test | Coverage |
|------|----------|
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
| TestRNN | RNN/LSTM Forward + Backpropagation Through Time |

---

*Last updated: 2026-03-05*
