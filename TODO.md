# ML_framework — TODO

Last updated: 2026-03-05

## Current progress (completed)
- Core `nn` Module/Parameter system and containers (`Sequential`, `ModuleDict`, etc).
- `Tensor` class with comprehensive API and native backpropagation support.
- Mathematical operations, reductions, broadcasting, and matrix multiplication.
- **Autograd Engine**: Fully functional `requires_grad`, `backward()`, tracking chain.
- **Dense Layers**: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softplus`, `Dropout` built natively on Tensors.
- **CNN Layers**: `Conv2d`, `MaxPool2d`, `AvgPool2d`, `ZeroPad2d`, `ConvTranspose2d` fully supporting backward pass.
- **Optimizers**: `SGD` (with momentum) and `Adam` (`optim.java`).
- **Loss Functions**: `cross_entropy_tensor`, `nll_loss`, `mse_loss_tensor`, `huber_loss` (in `nn.F`).
- **Normalization**: `BatchNorm1d`, `LayerNorm`, `InstanceNorm`.
- **System Optimizations**:
  - `DataLoader` with Multi-worker threading (Producer-Consumer logic).
  - Java Vector API (SIMD) integration for AVX2/AVX-512 acceleration.
- **Sequential Models**:
  - `RNNCell`, `LSTMCell`, `RNN`, `LSTM` with full BPTT support.
- **Examples**:
  - `TrainIris.java` (Simple Multi-class classification)
  - `TrainFashionMNIST.java` (using `DataLoader`)
  - `TrainCifar10.java` (using `DataLoader`)
- Comprehensive test suite (24 tests) including `TestRNN`.

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
