# ML_framework — TODO

Last updated: 2026-03-07

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
- **Phase 14: GPU Zero-Overhead Pipeline (NEW ✅)**:
  - **Kernel Fusion**: `Conv2d + Bias + ReLU` single-call execution.
  - **CUDA Streams**: Asynchronous Compute/Transfer pipelining.
  - **Arena Memory Pool**: `GpuMemoryPool` with auto-parameter detection.
  - **Custom PTX**: Native GPU kernels for Add/Sub/Mul (0 CPU Fallback).
  - **MemoryScope**: Automated ephemeral memory tracking and reset.
- **Model Training Features**:
  - `model.train()` and `model.eval()` module states.
  - `NN.Dropout(p)` with inverted dropout scaling capability.
- **Examples**:
  - `TrainIris.java` (Iris data)
  - `TrainFashionMNIST.java`
  - `TrainCifar10.java`
  - **`TrainSentiment.java`** (Real movie review dataset using LSTM)
- Comprehensive test suite (40 tests) including `TestBatch4`, `TestDropout`, and full GPU compatibility verification.
- ✅ **GPU Compatibility Audit**: Fully audited all mathematical and neural network operations for device-aware logic and automatic synchronization.

---

## Roadmap: Next Priorities

### Nhóm 1: Mở rộng Kiến trúc (Architectural Expansion)
1. **RNN & LSTM (HOÀN THÀNH ✅)**
2. **Transformer Blocks (PHẦN TIẾP THEO 🔲)**
   - Triển khai `MultiheadAttention`.
   - Nâng cấp `Softmax` hỗ trợ `dim`.
   - `TransformerEncoderLayer`.

### Nhóm 2: Tối ưu hóa Hệ thống (System Optimization) (HOÀN THÀNH ✅)
1. **DataLoader & Dataset API (HOÀN THÀNH ✅)**
2. **Vectorization (SIMD) (HOÀN THÀNH ✅)**
3. **GPU & cuDNN Optimization (HOÀN THÀNH ✅)**
   - ✅ **Phase 14**: Kernel Fusion, CUDA Streams, Arena Memory Pool, và Custom PTX Kernels.
   - ✅ **GPU Compatibility Audit**: Tự động đồng bộ hóa và nhận diện thiết bị.
4. **GPU Backward Kernels (PHẦN TIẾP THEO 🔲)**:
   - Chuyển đổi các hàm Backward sang cuDNN để giảm thiểu CPU sync.

---
**Steps to Begin:**
- Xây dựng **Transformer**: Cập nhật Softmax-dim và MultiheadAttention.
- Nâng cấp GPU: Chuyển đổi các hàm Backward (Grad) sang cuDNN.
