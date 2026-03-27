# ML Framework

[English](README.md) | [Tutorial](TUTORIAL.vn.md) | [Tutorial EN](TUTORIAL.md) | [API Reference](API_REFERENCE.vn.md) | [API Reference EN](API_REFERENCE.md)

![Java](https://img.shields.io/badge/Java-21+-orange)
![Build](https://img.shields.io/badge/Build-Gradle%20Multi--Module-blue)
![CUDA](https://img.shields.io/badge/GPU-JCuda%20%2B%20cuDNN-green)
![CPU](https://img.shields.io/badge/CPU-Vector%20API%20%2B%20OpenBLAS-purple)
![Tests](https://img.shields.io/badge/Tests-45%20registered-success)

Framework hoc may viet bang Java, lay cam hung tu PyTorch, phuc vu dong thoi 3 muc tieu: hoc cach deep learning hoat dong o muc framework, huan luyen mo hinh truc tiep trong Java, va mo rong dan tu CPU sang GPU bang JCuda, cuBLAS va cuDNN.

Repo hien da co tensor engine, autograd, he `Module/Parameter`, dataloader, optimizer, CNN, RNN, Transformer, mixed precision, OpenBLAS, custom CUDA kernels va bo test hoi quy dang pass toan bo.

## Getting Started

Neu ban chi muon bat dau that nhanh, chay dung 3 lenh nay:

```powershell
gradle wrapper
.\gradlew.bat :core:test
.\gradlew.bat :core:build
```

Sau do doc tiep:

- `TUTORIAL.vn.md` neu ban muon hoc theo tung buoc bang tieng Viet.
- `API_REFERENCE.vn.md` neu ban can ban do package va API chinh.

## So do tong quan

```mermaid
flowchart LR
    A[Data Loaders] --> B[Tensor / Torch]
    B --> C[Autograd Graph]
    C --> D[Module / Layers]
    D --> E[Optim / Scheduler]
    E --> P[Predict / Inference]
    B --> F[CPU Path]
    B --> G[GPU Path]
    F --> H[Vector API]
    F --> I[OpenBLAS]
    G --> J[JCuda / cuBLAS]
    G --> K[cuDNN]
    G --> L[PTX Kernels]
```

## Diem noi bat

- Tensor engine co reshape, broadcasting, indexing, reduction, transpose, gather/scatter, `matmul`, `bmm`.
- Autograd dynamic graph voi `requires_grad`, `grad_fn`, `backward()`, topological sort va version checking cho in-place ops.
- He `Module` kieu PyTorch voi `Sequential`, `ModuleList`, `ModuleDict`, `Parameter`.
- Layer cho nhieu bai toan pho bien: `Linear`, `Embedding`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, pooling, norm, attention, transformer encoder.
- CPU acceleration bang Java Vector API va OpenBLAS qua JavaCPP/bytedeco.
- GPU acceleration bang JCuda, cuBLAS, cuDNN, memory pool, CUDA streams, PTX kernels tuy bien, va theo doi VRAM voi `GpuMemoryMonitor`.
- Thu vien predict voi `Predictor`, `ImagePredictor`, `TextPredictor`, `BatchPredictor` va `PredictionPipeline` cho inference.
- Vi du end-to-end cho Iris, Fashion-MNIST, CIFAR-10, Sentiment Analysis, ViT, GAN, VAE — tat ca deu tich hop predict demo.
- Bo test PowerShell hien dang ky 45 test class va dang pass toan bo.

## Prediction / Inference

Package `predict` cung cap pipeline inference day du sau khi train:

```java
// Phan loai anh
ImagePredictor predictor = ImagePredictor.forCifar10(model);
PredictionResult result = predictor.predictFromPixels(imageData);
System.out.println(result); // -> airplane (0.9132), top-5

// Phan tich cam xuc
TextPredictor tp = TextPredictor.forSentiment(model, vocab, maxLen);
System.out.println(tp.predictSentiment("Great movie!")); // -> POSITIVE (0.92)

// Danh gia batch
BatchPredictor bp = new BatchPredictor(predictor);
float acc = bp.evaluateAccuracy(testLoader);
int[][] cm = bp.confusionMatrix(testLoader, 10);

// Fluent pipeline
PredictionPipeline.create(model)
    .loadWeights("model.bin")
    .labels(CIFAR10_LABELS)
    .topK(5)
    .predict(input);
```

## Benchmark tham khao

Cac so duoi day la ket qua do tren chinh repo hien tai bang benchmark san co. Day la so do dai dien, khong phai cam ket hieu nang tuyet doi vi con phu thuoc phan cung va moi truong.

| Tac vu | Duong chay | Kich thuoc | Ket qua do gan nhat |
|---|---|---|---|
| Matmul CPU lon | OpenBLAS | `256 x 256` | `0.58 ms / matmul` |
| Matmul CPU vectorized | Java Vector API | benchmark suite | `19.10 ms / matmul` |
| Regression suite | PowerShell runner | 45 test class | pass toan bo |

## Cong nghe chinh

| Thanh phan | Vai tro |
|---|---|
| Java 21 | Nen tang build va runtime |
| `jdk.incubator.vector` | SIMD cho phep toan CPU |
| JCuda / cuBLAS / cuDNN | Tang toc GPU cho tensor, matmul, conv, pooling, backward |
| OpenBLAS + JavaCPP | Tang toc `matmul` CPU kich thuoc lon |
| Gradle Kotlin DSL | Build, test, publish da nen tang |

## Yeu cau moi truong

### Bat buoc

- JDK 21 tro len. Repo hien da duoc kiem tra voi Temurin 21.0.10.
- Gradle 8+ (chi can khi chua sinh wrapper) hoac Gradle Wrapper.
- `java` trong `PATH`.

### Tuy chon nhung rat nen co

- NVIDIA GPU + CUDA driver neu muon dung duong GPU.
- CUDA toolkit neu muon build lai `kernels.cu` thanh `bin/kernels.ptx`.

### Dependency hien co trong repo

Thu muc `lib/` hien da chua cac JAR cho:

- JCuda
- JCublas
- JCudnn
- JavaCPP
- OpenBLAS

Trong phan lon truong hop, classpath `"bin;lib/*"` la du.

## Quick Start chi tiet

### 1. Sinh Gradle Wrapper (1 lan duy nhat)

```powershell
gradle wrapper
```

Neu ban da co wrapper day du (`gradle/wrapper/*`) thi bo qua buoc nay.

### 2. Build va test module core

Windows:

```powershell
.\gradlew.bat :core:clean :core:test :core:build
```

macOS/Linux:

```bash
./gradlew :core:clean :core:test :core:build
```

### 3. Build toan bo multi-module

Windows:

```powershell
.\gradlew.bat clean build
```

macOS/Linux:

```bash
./gradlew clean build
```

### 4. Legacy test runner (tuong thich voi script cu)

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
```

### 5. Script kiem tra CI/CD tu dong

- Chay nhanh (build + test + smoke test mot so example):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ci-test.ps1 -Mode quick
```

- Chay day du (smoke test toan bo example tim thay):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ci-test.ps1 -Mode full -ExampleTimeoutSec 60
```

## Lo trinh nen chay vi du

| Vi du | Muc tieu | Khi nao nen chay |
|---|---|---|
| `TrainIris` | Classification nho, de doc code | Bat dau o day |
| `TrainFashionMNIST` | Dataloader, mini-batch, MLP, GPU training | Sau Iris |
| `TrainSentiment` | NLP pipeline voi `Embedding` va LSTM | Khi muon xem text workflow |
| `TrainCifar10` | CNN tren du lieu anh that | Khi muon benchmark GPU |
| `TrainResNetCifar10` | Residual architecture | Sau khi quen CNN |
| `TrainViTCifar10` | Vision Transformer | Khi tim hieu attention |
| `TrainGANMnist` | Generative experiment | Khi muon mo rong nghien cuu |
| `TrainVAEMnist` | Variational autoencoder | Khi muon thu latent models |
| `TrainLeNet` | CNN co dien gon nhe | Khi can debug nhanh |
| `PredictDemo` | Demo thu vien predict day du | Khi muon hoc predict API |

## Cau truc repo

```text
src/com/user/nn/
  core/           Tensor, Torch, Functional, CUDAOps, GpuMemoryPool, MixedPrecision
  layers/         Linear, Conv, Embedding, Dropout, Bilinear
  activations/    ReLU, Sigmoid, Tanh, GELU, Softplus, Softmax, ...
  containers/     Sequential, ModuleList, ModuleDict, Flatten
  norm/           BatchNorm, LayerNorm, InstanceNorm, GroupNorm
  pooling/        MaxPool, AvgPool, AdaptiveAvgPool, ZeroPad
  attention/      MultiheadAttention, TransformerEncoderLayer
  rnn/            RNN, LSTM, GRU va cell tuong ung
  losses/         BCE, CrossEntropy, KLDiv, cosine, pairwise distance
  optim/          SGD, Adam, scheduler
  dataloaders/    Dataset, DataLoader, loader cho MNIST/CIFAR/Sentiment
  predict/        Predictor, ImagePredictor, TextPredictor, BatchPredictor, PredictionPipeline
  models/         Model hoan chinh cho NLP, CV va generative
  examples/       Chuong trinh train end-to-end voi predict demo

tests/
  java/com/user/nn/   Toan bo test Java
  run-tests.ps1       Script compile + chay regression suite
```

## Kien truc van hanh

### Tensor va autograd

`Tensor` la loi cua framework. Moi tensor giu shape, du lieu CPU, con tro GPU, `requires_grad`, gradient tich luy, `grad_fn` va version counter de phat hien in-place mutation pha graph.

### Device-aware execution

Framework dispatch theo thiet bi:

- tensor o CPU thi dung CPU path
- tensor o GPU thi uu tien GPU path
- phep toan lon tren CPU co the dung OpenBLAS
- phep toan GPU dung cuBLAS, cuDNN hoac PTX kernel tuy bien
- neu GPU khong kha dung thi fallback ve CPU khi co the

### Memory management

Repo hien co 3 tang quan ly bo nho quan trong:

- `AutoCloseable` tren `Tensor`
- `Cleaner` thay cho `finalize()` cho GPU memory safety net
- `MemoryScope` + `GpuMemoryPool` de giam overhead VRAM trong training loop

## Tai lieu di kem

- `TUTORIAL.vn.md`: huong dan tung buoc bang tieng Viet
- `TUTORIAL.md`: tutorial tieng Anh
- `API_REFERENCE.vn.md`: package reference tieng Viet
- `API_REFERENCE.md`: package reference tieng Anh

## Trang thai phat hanh

- Build mac dinh da chuyen sang Gradle multi-module (`:core`, `:examples`, `:tests`).
- Da sinh day du Gradle Wrapper (`gradlew`, `gradlew.bat`, `gradle/wrapper/*`).
- Lenh kiem tra release da xac nhan:

```powershell
.\gradlew.bat :core:clean :core:test :core:classes --no-daemon
```

- Cach chay cac example thong nhat va de nhat qua Gradle:

```powershell
.\gradlew.bat "-PmainClass=com.user.nn.examples.TrainFashionMNIST" :examples:run --no-daemon
```

## Ghi chu thuc te

- Mot so vi du tu tai du lieu neu thieu, mot so khac dung du lieu da co san trong `data/`.
- Luong build mac dinh da chuyen sang Gradle va ho tro da nen tang.
- Neu chay `java -cp` thu cong tren Windows, classpath dung dau `;`.
- Neu sua `src/com/user/nn/core/kernels.cu`, ban can build lai PTX tuong ung.

---

Tai lieu duoc cap nhat theo trang thai code hien tai vao 2026-03-18.
