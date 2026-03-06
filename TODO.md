# ML_framework — TODO

Last updated: 2026-03-05

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
- **Normalization**: `BatchNorm1d`, `LayerNorm`, `InstanceNorm`, **`GroupNorm`**.
- **Similarity & Distance**: `CosineSimilarity`, `PairwiseDistance`.
- **NLP Utilities**: **`Embedding`**, `Vocabulary`, `BasicTokenizer`.
- **System Optimizations**:
  - `DataLoader` with Multi-worker threading.
  - Java Vector API (SIMD) integration for AVX2/AVX-512 acceleration.
- **Examples**:
  - `TrainIris.java` (Iris data)
  - `TrainFashionMNIST.java`
  - `TrainCifar10.java`
  - **`TrainSentiment.java`** (Real movie review dataset using LSTM)
- Comprehensive test suite (31 tests) including `TestBatch4`.

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
   - Matmul tối ưu vượt trội so với for-loop truyền thống.

---
**Steps to Begin:**
- Chuyển sang phần **Transformer**: Cập nhật Softmax-dim và MultiheadAttention.
