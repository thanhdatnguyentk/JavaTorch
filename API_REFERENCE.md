# Package API Reference

[Tieng Viet](API_REFERENCE.vn.md) | [README](README.md) | [Tutorial](TUTORIAL.md)

This document is not a full JavaDoc replacement. Its purpose is to give you a package-level map of the framework so you can quickly find the right part of the codebase.

## `com.user.nn.core`

- `Tensor`: tensor storage, device state, gradients, backward, memory lifecycle.
- `Torch`: tensor ops, reductions, broadcasting, matmul, and initialization helpers.
- `Functional`: functional-style losses and utility ops.
- `CUDAOps`: JCuda, cuBLAS, cuDNN, and PTX kernel wrappers.
- `GpuMemoryPool`: VRAM pool.
- `MemoryScope`: scoped lifecycle management.
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
- `BatchNorm2d`
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
- `KLDivLoss`
- `L1Loss`
- `CosineSimilarity`
- `PairwiseDistance`

## `com.user.nn.optim`

- `Optim`: `SGD`, `Adam`
- `Scheduler`: schedulers such as `StepLR`

## `com.user.nn.dataloaders`

- `Data`: `Dataset`, `DataLoader`, `Vocabulary`, `BasicTokenizer`
- `MnistLoader`
- `Cifar10Loader`
- `MovieCommentLoader`

## `com.user.nn.metrics`

- `Metric`
- `Accuracy`
- `MeanAbsoluteError`
- `MeanSquaredError`
- `MetricTracker`
- `Evaluator`

## `com.user.nn.models`

- `SentimentModel`
- `models.cv`: `LeNet`, `VGG`, `ResNet`, `ViT`
- `models.generative`: `GAN`, `VAE`

## `com.user.nn.examples`

- `TrainIris`
- `TrainFashionMNIST`
- `TrainCifar10`
- `TrainResNetCifar10`
- `TrainSentiment`
- `TrainViTCifar10`
- `TrainGANMnist`
- `TrainVAEMnist`
- `TrainLeNet`

## Quick lookup guide

- Want to add a new op: start in `core`
- Want to build a new model: read `layers`, `containers`, `norm`, `pooling`
- Want to train on real datasets: read `dataloaders`, `optim`, `metrics`, `examples`
- Want to optimize performance: read `CUDAOps`, `BlasOps`, `GpuMemoryPool`, `kernels.cu`