# ML_framework — TODO

Last updated: 2026-03-06

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
   - Triển khai `RNNCell`, `LSTMCell`, `RNN`, `LSTM`.
   - Hỗ trợ BPTT tự động qua Autograd.
2. **Transformer Blocks (PHẦN TIẾP THEO 🔲)**
   - Triển khai `MultiheadAttention` (Linear Q,K,V + Scaled Dot Product).
   - Nâng cấp `Softmax` hỗ trợ `dim` tùy chọn.
   - Xây dựng module `TransformerEncoderLayer`.

### Nhóm 2: Tối ưu hóa Hệ thống (System Optimization) (HOÀN THÀNH ✅)
1. **DataLoader & Dataset API (HOÀN THÀNH ✅)**
   - Interface `Dataset` và `DataLoader` đa luồng.
2. **Vectorization (SIMD) (HOÀN THÀNH ✅)**
   - Tích hợp **Java Vector API**.
3. **GPU & cuDNN (HOÀN THÀNH ✅)**
   - Tích hợp **JCublas** cho matmul và **JCudnn** cho Conv/Pool/ReLU.
   - Cơ chế hybrid dispatch (GPU forward / CPU backward sync).
   - ✅ **GPU Compatibility Audit**: Kiểm tra và cập nhật toàn bộ hàm toán học/layer đảm bảo tự động đồng bộ hóa và nhận diện thiết bị (device-aware).

---
**Steps to Begin:**
- Xây dựng **Transformer**: Cập nhật Softmax-dim và MultiheadAttention.
- Tối ưu hóa GPU: Chuyển đổi các hàm Backward (Grad) sang cuDNN để giảm thiểu việc đồng bộ hóa với CPU.
