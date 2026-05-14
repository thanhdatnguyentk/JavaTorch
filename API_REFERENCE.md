# Package API Reference

[Tieng Viet](API_REFERENCE.vn.md) | [README](README.md) | [Tutorial](TUTORIAL.md)

This document is not a full JavaDoc replacement. Its purpose is to give you a package-level map of the framework so you can quickly find the right part of the codebase.

## `com.user.nn.core`

- `Tensor`: tensor storage, device state, gradients, backward, memory lifecycle.
- `Torch`: tensor ops, reductions, broadcasting, matmul, and initialization helpers (~3300 lines).
- `Functional`: functional-style losses and utility ops.
- `CUDAOps`: JCuda, cuBLAS, cuDNN, and PTX kernel wrappers.
- `GpuMemoryPool`: VRAM pool with auto-expanding when batch demand exceeds initial allocation.
- `GpuMemoryMonitor`: live VRAM usage tracking and fallback allocation reporting.
- `MemoryScope`: scoped lifecycle management (triggers pool auto-expand on `close()`).
- `MixedPrecision`: mixed-precision support.
- `Module`: base abstraction for layers and models.
- `Parameter`: gradient-carrying parameter wrapper.
- `NN`: low-level matrix helpers.
- `BlasOps`: OpenBLAS wrapper for large CPU matmul.

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
- `BatchNorm2d` (CPU + GPU, with full autograd backward)
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
- `FocalLoss` (configurable α/γ, binary and multi-class)
- `KLDivLoss`
- `L1Loss`
- `CosineSimilarity`
- `PairwiseDistance`

## `com.user.nn.optim`

- `Optim`: `SGD` (with momentum), `Adam`
- `Scheduler`: learning-rate schedulers such as `StepLR`

## `com.user.nn.dataloaders`

- `Data`: `Dataset`, `DataLoader`, `Vocabulary`, `BasicTokenizer`
- `MnistLoader`
- `Cifar10Loader`
- `MovieCommentLoader`
- `AnimeFaceLoader`
- `UitVsfcLoader` (Vietnamese sentiment & topic classification)

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
- `MultiTaskLSTMModel` (multi-head sentiment + topic)
- `MultiTaskTransformerModel` (Transformer-based multi-task)
- `models.cv`: `LeNet`, `VGG`, `ResNet`, `ViT`, `YOLO`, `SSD`, `RetinaNet`, `FasterRCNN`, `RPN`
- `models.generative`: `GAN`, `VAE`

## `com.user.nn.examples`

### Training Examples
- `TrainIris` — Iris classification (beginner)
- `TrainLeNet` — Classic LeNet CNN
- `TrainFashionMNIST` — Fashion-MNIST with CNN + GPU
- `TrainCifar10` — CIFAR-10 classification
- `TrainResNetCifar10` — ResNet-18 on CIFAR-10
- `TrainViTCifar10` — Vision Transformer on CIFAR-10
- `TrainSentiment` — Movie review sentiment analysis (LSTM)
- `TrainUitVsfc` — UIT-VSFC Vietnamese sentiment classification
- `TrainUitVsfcMultitask` — UIT-VSFC multi-task (sentiment + topic, LSTM vs Transformer)
- `TrainGANMnist` — GAN on MNIST
- `TrainGANAnime` — GAN on anime faces
- `TrainVAEMnist` — VAE on MNIST
- `TrainYOLOCoco` — YOLO on COCO
- `TrainAllDetectorsCoco` — All 4 detection models on COCO

### Benchmark Examples
- `BenchmarkResNetCifar10` — ResNet benchmark (JavaTorch)
- `BenchmarkSentiment` — Sentiment benchmark (JavaTorch)
- `BenchmarkDl4jResNetCifar10` — ResNet benchmark (DL4J baseline)
- `BenchmarkDl4jSentiment` — Sentiment benchmark (DL4J baseline)
- `BenchmarkMemoryPool` — GPU memory pool benchmark

### Demo & Utility
- `PredictDemo` — Full predict API demo
- `ObjectDetectionDemo` — Object detection models demo
- `ProgressAndVisualizationDemo` — Progress bar and visualization demo

## Quick lookup guide

- Want to add a new op: start in `core`
- Want to build a new model: read `layers`, `containers`, `norm`, `pooling`
- Want to train on real datasets: read `dataloaders`, `optim`, `metrics`, `examples`
- Want to optimize performance: read `CUDAOps`, `BlasOps`, `GpuMemoryPool`, `kernels.cu`
- Want to do inference: read `predict`