# API Reference theo package

[English](API_REFERENCE.md) | [README](README.vn.md) | [Tutorial](TUTORIAL.vn.md)

Tài liệu này không thay thế JavaDoc chi tiết theo từng method. Mục tiêu của nó là cho bạn bản đồ package-level để biết nên đọc và dùng phần nào của framework trước.

## `com.user.nn.core`

Package lõi của framework.

- `Tensor`: tensor, gradient, backward, lifecycle.
- `Torch`: tensor ops, math ops, reductions, broadcasting, matmul, init helpers.
- `Functional`: functional losses và utility ops.
- `CUDAOps`: wrapper JCuda, cuBLAS, cuDNN, PTX kernels.
- `GpuMemoryPool`: VRAM pool.
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
- `Scheduler`: learning-rate schedulers như `StepLR`

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

## Gợi ý tra cứu nhanh

- Muốn viết op mới: đọc `core`
- Muốn dựng model mới: đọc `layers`, `containers`, `norm`, `pooling`
- Muốn train trên dữ liệu thật: đọc `dataloaders`, `optim`, `metrics`, `examples`
- Muốn tăng tốc: đọc `CUDAOps`, `BlasOps`, `GpuMemoryPool`, `kernels.cu`
