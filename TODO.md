# ML_framework — TODO

Last updated: 2026-03-09

## Release readiness (2026-03)
- Gradle multi-module build is in place (`:core`, `:examples`, `:tests`).
- Gradle Wrapper has been generated (`gradlew`, `gradlew.bat`, `gradle/wrapper/*`).
- Verified command for release checks:
  - `./gradlew :core:clean :core:test :core:build --no-daemon`
- Legacy script path remains available for compatibility:
  - `tests/run-tests.ps1`

## Current progress (completed)
- Core `nn` Module/Parameter system and containers (`Sequential`, `ModuleDict`, etc).
- `Tensor` class with comprehensive API and native backpropagation support.
- Mathematical operations, reductions, broadcasting, and matrix multiplication.
- **Autograd Engine**: Fully functional `requires_grad`, `backward()` iterative topological sort, tracking chain.
- **Dense Layers**: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softplus`, `Dropout`, and **`Bilinear`**.
- **CNN Layers**: `Conv1d`, `Conv2d`, `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `ZeroPad2d`, `ConvTranspose2d`.
- **RNN/LSTM**: `RNNCell`, `LSTMCell`, `GRUCell`, `RNN`, `LSTM`, `GRU` with full BPTT support.
- **Optimizers**: `SGD` (with momentum) and `Adam`.
- **Loss Functions**: `cross_entropy`, `nll_loss`, `mse_loss`, `huber_loss`, `BCELoss`, `BCEWithLogitsLoss`, `KLDivLoss`, `L1Loss`.
- **Normalization**: `BatchNorm1d` (Train/Eval aware), `LayerNorm`, `InstanceNorm`, **`GroupNorm`**.
- **Similarity & Distance**: `CosineSimilarity`, `PairwiseDistance`.
- **NLP Utilities**: **`Embedding`**, `Vocabulary`, `BasicTokenizer`.
- **System Optimizations**:
  - `DataLoader` with Multi-worker threading.
  - Java Vector API (SIMD) integration (AVX2/AVX-512).
  - **GPU Acceleration**: ✅ JCuda + JCublas + JCudnn (Conv2d, MaxPool2d, ReLU).
- ✅ **Phase 14: GPU Zero-Overhead Pipeline (NEW ✅)**:
  - **Kernel Fusion**: `Conv2d + Bias + ReLU` single-call execution.
  - **CUDA Streams**: Asynchronous Compute/Transfer pipelining.
  - **Arena Memory Pool**: `GpuMemoryPool` with auto-parameter detection.
  - **Custom PTX**: Native GPU kernels for Add/Sub/Mul (0 CPU Fallback).
  - **MemoryScope**: Automated ephemeral memory tracking and reset.
- **Phase 16: Transformers & Vision Transformer (ViT) (NEW ✅)**:
  - **MultiheadAttention**: Full implementation with batched matrix multiplication (`bmm`).
  - **TransformerEncoderLayer**: Pre-norm architecture with LayerNorm and FeedForward.
  - **ViT (Vision Transformer)**: Complete architecture with patch embedding, learnable tokens, and positional embeddings.
  - **ND Transpose & Expand**: Generalized tensor operations to support Transformer rank-4 shapes.
  - **Robust reduceSumToShape**: Fixed broadcasting bugs in the autograd engine.
- **Model Training Features**:
  - `model.train()` and `model.eval()` module states.
  - `NN.Dropout(p)` with inverted dropout scaling capability.
- **Examples**:
  - `TrainIris.java` (Iris data)
  - `TrainFashionMNIST.java`
  - `TrainCifar10.java`
  - **`TrainSentiment.java`** (Real movie review dataset using LSTM)
  - **`TrainViTCifar10.java`** (End-to-end Vision Transformer Training on GPU)
  - **`TestViT.java`** (Functional verification of ViT on GPU)
- Comprehensive test suite (40 tests) including `TestBatch4`, `TestDropout`, and full GPU compatibility verification.
- ✅ **Phase 15: Computer Vision Deepening (NEW ✅)**:
  - **Tensor.cat & Tensor.narrow**: Fully implemented with Autograd and GPU acceleration.
  - **VGG**: Configurable VGG11-19 architectures with optional BatchNorm.
  - **ResNet**: ResNet-18/34 with `BasicBlock` and skip connection support (Torch.add autograd).
  - **Global Avg Pooling**: `adaptive_avg_pool2d` implemented in `NN.F`.
  - **Evaluator**: Centralized evaluation class using DataLoader.
- ✅ **GPU Compatibility Audit**: Fully audited all mathematical and neural network operations for device-aware logic and automatic synchronization.
- ✅ **GPU Conv2d Backward**: Full cuDNN implementation for BackwardData, BackwardFilter, and BackwardBias.
- ✅ **Phase 4: Generative Models & Advanced GPU Ops (NEW ✅)**:
- **GAN & VAE**: Full implementations with GPU-accelerated training loops.
- **GPU Activations**: ReLU, LeakyReLU, Sigmoid, Tanh kernels (Forward/Backward).
- **GPU BCE Loss**: Custom kernels for Binary Cross Entropy (logits and probs).
- **Differentiable Reductions**: `sum_tensor` and `mean_tensor` with full GPU autograd.
- **Serialization (Checkpoints)**: ✅ Implemented `model.save()` and `model.load()` using binary DataStreams.
- **LR Schedulers**: ✅ Added `Scheduler` system with `StepLR`.

---

## Roadmap: Next Priorities

### Nhóm 1: Mở rộng Kiến trúc (Architectural Expansion)
1. **Transformer Blocks (HOÀN THÀNH ✅)**
2. **Vision Transformer (ViT) (HOÀN THÀNH ✅)**
3. **Generative Models (HOÀN THÀNH ✅)**:
   - Triển khai **GAN** (Generative Adversarial Networks) trên MNIST.
   - **VAE** (Variational Autoencoders) trên MNIST.

### Nhóm 2: Tối ưu hóa Hệ thống (System Optimization)
1. **GPU MaxPool Backward**: Chuyển đổi pooling backward sang cuDNN.
2. **GPU Transpose (ND)**: Implement ND transpose kernel to avoid CPU synchronization for 3D/4D tensors.
3. **Automated Mixed Precision (AMP)**: Support FP16 training to save VRAM and increase speed on Tensor Cores.

---
**Steps to Begin:**
- Xây dựng **Generative Models**: Bắt đầu với GAN trên MNIST.
- Nâng cấp GPU: Tiếp tục chuyển đổi các hàm Backward còn lại sang cuDNN.
