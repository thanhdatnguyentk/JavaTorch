# ML_framework — TODO

Last updated: 2026-05-11

## Current Status

- ✅ Gradle multi-module build in place (`:core`, `:examples`, `:tests`).
- ✅ Gradle Wrapper generated (`gradlew`, `gradlew.bat`, `gradle/wrapper/*`).
- ✅ 50 test files, 118+ test methods, 100% JUnit 5, 0 `main()` runners.
- ✅ Full CPU test pass + GPU tests pass with `-PincludeGPU=true`.

## Completed Features

### Core Engine
- ✅ `Tensor` class with comprehensive API and native backpropagation.
- ✅ Mathematical operations, reductions, broadcasting, and matrix multiplication.
- ✅ **Autograd Engine**: `requires_grad`, `backward()`, iterative topological sort, version checking.
- ✅ Java Vector API (SIMD) integration (AVX2/AVX-512).
- ✅ OpenBLAS CPU matmul via JavaCPP/bytedeco.

### Layers & Modules
- ✅ **Dense Layers**: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `GELU`, `ELU`, `SiLU`, `Softplus`, `Dropout`, `Bilinear`.
- ✅ **CNN Layers**: `Conv1d`, `Conv2d`, `ConvTranspose2d`, `MaxPool1d/2d`, `AvgPool1d/2d`, `AdaptiveAvgPool2d`, `ZeroPad2d`.
- ✅ **RNN/LSTM/GRU**: `RNNCell`, `LSTMCell`, `GRUCell`, `RNN`, `LSTM`, `GRU` with full BPTT.
- ✅ **Normalization**: `BatchNorm1d`, `BatchNorm2d` (CPU + GPU with autograd), `LayerNorm`, `InstanceNorm`, `GroupNorm`.
- ✅ **Attention**: `MultiheadAttention`, `TransformerEncoderLayer`.
- ✅ **NLP**: `Embedding`, `Vocabulary`, `BasicTokenizer`.

### Losses & Optimizers
- ✅ **Losses**: `CrossEntropy`, `NLL`, `MSE`, `Huber`, `BCE`, `BCEWithLogits`, `KLDiv`, `L1`, `FocalLoss`.
- ✅ **Similarity/Distance**: `CosineSimilarity`, `PairwiseDistance`.
- ✅ **Optimizers**: `SGD` (with momentum) and `Adam`.
- ✅ **Schedulers**: `StepLR`.

### GPU Acceleration
- ✅ JCuda + JCublas + JCudnn (Conv2d, MaxPool2d, ReLU, Embedding).
- ✅ **Kernel Fusion**: `Conv2d + Bias + ReLU` single-call execution.
- ✅ **CUDA Streams**: Asynchronous Compute/Transfer pipelining.
- ✅ **Arena Memory Pool**: `GpuMemoryPool` with auto-expanding.
- ✅ **Custom PTX**: Native GPU kernels for Add/Sub/Mul.
- ✅ **MemoryScope**: Automated ephemeral memory tracking and reset.
- ✅ **GPU Conv2d Backward**: Full cuDNN BackwardData/BackwardFilter/BackwardBias.
- ✅ **GPU Activations**: ReLU, LeakyReLU, Sigmoid, Tanh kernels (Forward/Backward).
- ✅ **GPU BCE Loss**: Custom kernels for Binary Cross Entropy.

### Models
- ✅ **CV**: LeNet, VGG (11-19), ResNet (18/34), ViT.
- ✅ **Object Detection**: YOLO v1, SSD (300/512), RetinaNet (FPN + Focal Loss), Faster R-CNN (RPN + ROI Pooling).
- ✅ **Generative**: GAN, VAE.
- ✅ **NLP**: SentimentModel, MultiTaskLSTMModel, MultiTaskTransformerModel.

### Training Examples (28 files)
- ✅ `TrainIris`, `TrainLeNet`, `TrainFashionMNIST`, `TrainCifar10`, `TrainResNetCifar10`.
- ✅ `TrainViTCifar10`, `TrainSentiment`.
- ✅ `TrainUitVsfc`, `TrainUitVsfcMultitask` (LSTM vs Transformer comparison).
- ✅ `TrainGANMnist`, `TrainGANAnime`, `TrainVAEMnist`.
- ✅ `TrainYOLOCoco`, `TrainAllDetectorsCoco`.
- ✅ Benchmarks: ResNet, Sentiment, DL4J baselines, MemoryPool.

### Infrastructure
- ✅ `Module/Parameter` system with `Sequential`, `ModuleList`, `ModuleDict`.
- ✅ `DataLoader` with multi-worker threading.
- ✅ Model serialization (`save()`/`load()`).
- ✅ `train()` / `eval()` mode switching (Dropout, BatchNorm).
- ✅ Prediction library: `Predictor`, `ImagePredictor`, `TextPredictor`, `BatchPredictor`, `PredictionPipeline`.
- ✅ Web Dashboard (Vue 3 + WebSocket) with real-time metrics.
- ✅ Progress bars, visualization, training history tracking.
- ✅ CI/CD scripts (`ci-test.ps1`, `run-benchmark-matrix.ps1`).

---

## Roadmap: Next Priorities

### Nhóm 1: Tối ưu hóa Hệ thống (System Optimization)
1. **Automated Mixed Precision (AMP)**: Support FP16 training to save VRAM and increase speed on Tensor Cores.
2. **GPU ND Transpose**: Implement ND transpose kernel to avoid CPU synchronization for 3D/4D tensors.
3. **cuDNN RNN**: Migrate LSTM/GRU forward/backward to cuDNN for GPU acceleration.

### Nhóm 2: Mở rộng Kiến trúc (Architectural Expansion)
1. **Transformer Decoder**: Add decoder blocks and seq2seq support.
2. **Advanced Object Detection**: YOLOv3+, Mask R-CNN variants.
3. **Data Augmentation**: RandomCrop, RandomFlip, ColorJitter for detection and classification.
4. **Learning Rate Schedulers**: CosineAnnealingLR, ReduceLROnPlateau.

### Nhóm 3: Chất lượng (Quality)
1. **Expand test coverage**: Add tests for object detection models, multi-task models.
2. **Evaluation metrics**: mAP calculation for detection, macro-F1 for multi-task.
3. **Documentation**: Full JavaDoc generation.

---

**Steps to Begin:**
- Nâng cấp GPU: Tiếp tục chuyển đổi RNN/LSTM backward sang cuDNN.
- AMP: Thêm FP16 tensor support cho memory-intensive models.
- Test coverage: Viết thêm unit tests cho models và losses.
