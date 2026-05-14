# API Reference theo package

[English](API_REFERENCE.md) | [README](README.vn.md) | [Tutorial](TUTORIAL.vn.md)

Tài liệu này không thay thế JavaDoc chi tiết theo từng method. Mục tiêu của nó là cho bạn bản đồ package-level để biết nên đọc và dùng phần nào của framework trước.

## `com.user.nn.core`

Package lõi của framework.

- `Tensor`: tensor, gradient, backward, lifecycle.
- `Torch`: tensor ops, math ops, reductions, broadcasting, matmul, init helpers (~3300 lines).
- `Functional`: functional losses và utility ops.
- `CUDAOps`: wrapper JCuda, cuBLAS, cuDNN, PTX kernels.
- `GpuMemoryPool`: VRAM pool tự động mở rộng.
- `GpuMemoryMonitor`: theo dõi VRAM usage thời gian thực.
- `MemoryScope`: quản lý tensor tạm theo scope.
- `MixedPrecision`: mixed precision.
- `Module`: base abstraction cho model/layer.
- `Parameter`: wrapper tham số có gradient.
- `NN`: low-level matrix helpers.
- `BlasOps`: wrapper OpenBLAS cho CPU matmul lớn.

## `com.user.nn.layers`

- `Linear`
- `Bilinear`
- `Embedding`
- `Conv1d`
- `Conv2d`
- `ConvTranspose2d`
- `Dropout`
- `Flatten`
- `ROIPooling`

## `com.user.nn.activations`

- `ReLU`
- `Sigmoid`
- `Tanh`
- `LeakyReLU`
- `GELU`
- `ELU`
- `SiLU`
- `Softplus`
- `Softmax`
- `LogSoftmax`

## `com.user.nn.containers`

- `Sequential`
- `ModuleList`
- `ModuleDict`
- `Flatten`

## `com.user.nn.norm`

- `BatchNorm1d`
- `BatchNorm2d` (CPU + GPU, hỗ trợ autograd backward đầy đủ)
- `LayerNorm`
- `InstanceNorm`
- `GroupNorm`

## `com.user.nn.pooling`

- `MaxPool1d`
- `MaxPool2d`
- `AvgPool1d`
- `AvgPool2d`
- `AdaptiveAvgPool2d`
- `ZeroPad2d`

## `com.user.nn.attention`

- `MultiheadAttention`
- `TransformerEncoderLayer`

## `com.user.nn.rnn`

- `RNNCell`, `RNN`
- `LSTMCell`, `LSTM`
- `GRUCell`, `GRU`

## `com.user.nn.losses`

- `BCELoss`
- `BCEWithLogitsLoss`
- `CrossEntropyLoss`
- `FocalLoss` (α/γ cấu hình được, hỗ trợ binary và multi-class)
- `KLDivLoss`
- `L1Loss`
- `CosineSimilarity`
- `PairwiseDistance`

## `com.user.nn.optim`

- `Optim`: `SGD` (có momentum), `Adam`
- `Scheduler`: learning-rate schedulers như `StepLR`

## `com.user.nn.dataloaders`

- `Data`: `Dataset`, `DataLoader`, `Vocabulary`, `BasicTokenizer`
- `MnistLoader`
- `Cifar10Loader`
- `MovieCommentLoader`
- `AnimeFaceLoader`
- `UitVsfcLoader` (phân loại cảm xúc & chủ đề tiếng Việt)

## `com.user.nn.metrics`

- `Metric`
- `Accuracy`
- `MeanAbsoluteError`
- `MeanSquaredError`
- `MetricTracker`
- `Evaluator`

## `com.user.nn.predict`

- `Predictor`
- `ImagePredictor`
- `TextPredictor`
- `BatchPredictor`
- `PredictionResult`
- `PredictionPipeline`

## `com.user.nn.models`

- `SentimentModel`
- `MultiTaskLSTMModel` (đa nhiệm sentiment + topic)
- `MultiTaskTransformerModel` (Transformer đa nhiệm)
- `models.cv`: `LeNet`, `VGG`, `ResNet`, `ViT`, `YOLO`, `SSD`, `RetinaNet`, `FasterRCNN`, `RPN`
- `models.generative`: `GAN`, `VAE`

## `com.user.nn.examples`

### Ví dụ huấn luyện
- `TrainIris` — Phân loại Iris (người mới bắt đầu)
- `TrainLeNet` — LeNet CNN cổ điển
- `TrainFashionMNIST` — Fashion-MNIST với CNN + GPU
- `TrainCifar10` — Phân loại CIFAR-10
- `TrainResNetCifar10` — ResNet-18 trên CIFAR-10
- `TrainViTCifar10` — Vision Transformer trên CIFAR-10
- `TrainSentiment` — Phân tích cảm xúc movie review (LSTM)
- `TrainUitVsfc` — Phân loại cảm xúc UIT-VSFC tiếng Việt
- `TrainUitVsfcMultitask` — Đa nhiệm UIT-VSFC (sentiment + topic, LSTM vs Transformer)
- `TrainGANMnist` — GAN trên MNIST
- `TrainGANAnime` — GAN trên anime faces
- `TrainVAEMnist` — VAE trên MNIST
- `TrainYOLOCoco` — YOLO trên COCO
- `TrainAllDetectorsCoco` — Tất cả 4 detection models trên COCO

### Benchmark
- `BenchmarkResNetCifar10`, `BenchmarkSentiment` — JavaTorch
- `BenchmarkDl4jResNetCifar10`, `BenchmarkDl4jSentiment` — DL4J baseline
- `BenchmarkMemoryPool` — GPU memory pool

### Demo
- `PredictDemo` — Demo predict API
- `ObjectDetectionDemo` — Demo object detection
- `ProgressAndVisualizationDemo` — Demo progress bar và visualization

## Gợi ý tra cứu nhanh

- Muốn viết op mới: đọc `core`
- Muốn dựng model mới: đọc `layers`, `containers`, `norm`, `pooling`
- Muốn train trên dữ liệu thật: đọc `dataloaders`, `optim`, `metrics`, `examples`
- Muốn tăng tốc: đọc `CUDAOps`, `BlasOps`, `GpuMemoryPool`, `kernels.cu`
- Muốn inference: đọc `predict`
