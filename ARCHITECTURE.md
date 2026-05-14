# JavaTorch Testing & Automation Architecture

Tài liệu này mô tả kiến trúc hệ thống kiểm thử và quy trình tự động hóa của dự án JavaTorch. Mục tiêu là đảm bảo hệ thống nhanh (Fast CI), sâu (GPU validation) và đáng tin cậy (Benchmarks).

Cập nhật lần cuối: 2026-05-11.

## 1. Nguyên tắc cốt lõi (Core Principles)

- **Tách biệt mục đích**: Unit test chạy mọi lúc; GPU test và Benchmark chỉ chạy khi có môi trường phù hợp.
- **Độc lập với phần cứng**: Code core (Tensor logic) phải chạy được trên CPU để CI luôn xanh.
- **Mọi thứ qua Gradle**: Toàn bộ vòng đời từ build, test đến benchmark đều được quản lý tập trung.
- **Tính nhất quán (Determinism)**: Mọi test case liên quan đến Random/Weights phải được fix seed (`Torch.manual_seed(42)`).

## 2. Cấu trúc thư mục (Project Structure)

```
JavaTorch/
├── core/
│   ├── build.gradle.kts          # Library build, dependencies, PTX compilation
│   └── src/test/java/            # (Legacy) Unit Tests cho module core
│
├── tests/
│   ├── build.gradle.kts          # JUnit 5 config, GPU tag exclusion
│   └── java/com/user/nn/        # 50 test files (100% JUnit 5)
│       ├── TestTensor.java       # Tensor construction, reshape, ops
│       ├── TestAutogradSimple.java, TestAutogradMatmul.java, etc.
│       ├── TestBatchNorm2d.java  # BatchNorm2d CPU + backward regression
│       ├── TestConvPool.java     # Conv2d + MaxPool2d verification
│       ├── TestGPUKernels.java   # GPU kernel tests (@Tag("gpu"))
│       ├── TestGPUBenchmark.java # GPU performance (@Tag("gpu"))
│       └── ... (50 files total)
│
├── examples/                     # Training examples & benchmarks
│   ├── build.gradle.kts          # JavaExec tasks
│   └── src/                      # 28 example/benchmark files
│
├── src/com/user/nn/             # ALL source code (consumed by :core)
│   ├── core/                    # Tensor, Torch, Module, CUDAOps, MemoryScope
│   ├── layers/                  # Linear, Conv2d, Embedding, Dropout, etc.
│   ├── activations/             # ReLU, GELU, SiLU, Softmax, etc.
│   ├── containers/              # Sequential, ModuleList, ModuleDict
│   ├── norm/                    # BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm
│   ├── pooling/                 # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   ├── rnn/                     # LSTM, GRU, RNN
│   ├── attention/               # MultiheadAttention, TransformerEncoderLayer
│   ├── losses/                  # CrossEntropy, BCE, FocalLoss, KLDiv, etc.
│   ├── optim/                   # SGD, Adam, StepLR
│   ├── metrics/                 # Accuracy, MSE, MAE, Evaluator
│   ├── dataloaders/             # MNIST, CIFAR-10, UIT-VSFC, MovieComment, AnimeFace
│   ├── predict/                 # Predictor, ImagePredictor, TextPredictor
│   ├── models/                  # SentimentModel, MultiTask, CV, Generative
│   └── utils/                   # Dashboard, Progress, Visualization
│
├── build.gradle.kts             # Root multi-module config
└── ARCHITECTURE.md              # This file
```

## 🧪 3. Chiến lược kiểm thử 4 tầng (4-Tier Strategy)

### 3.1. Unit Tests (Mặc định - Fast CI)
- **Mục tiêu**: Kiểm tra các phép toán Tensor đơn lẻ, tính toán đạo hàm, khởi tạo layer.
- **Ràng buộc**: Chạy cực nhanh (<1s/test), không yêu cầu GPU.
- **Yêu cầu**: Sai số số thực (Delta) tối đa `1e-5`.
- **Số lượng hiện tại**: ~90 test methods trên CPU.

### 3.2. Integration Tests
- **Mục tiêu**: Kiểm tra luồng dữ liệu xuyên suốt (Model Training loop, Forward/Backward pass).
- **Tags**: `@Tag("integration")`.

### 3.3. GPU Tests
- **Mục tiêu**: Kiểm tra CUDA kernels và tính đúng đắn trên phần cứng tăng tốc.
- **Cơ chế**: GPU guard check trước mỗi test.
- **Tags**: `@Tag("gpu")`, `@Tag("gpu-smoke")`, `@Tag("gpu-nightly")`.
- **Mặc định bị bỏ qua** trừ khi chạy với `-PincludeGPU=true`.

### 3.4. Benchmarks
- **Mục tiêu**: Đo lường hiệu năng (Ops/sec, Memory usage).
- **Chạy bằng**: Gradle JavaExec tasks trong `:examples` module.

## ⚙️ 4. Cấu hình Gradle (Execution Control)

### Các Task thực thi chính

| Task | Lệnh thực thi | Mô tả |
| :--- | :--- | :--- |
| Test Suite (CPU) | `./gradlew :tests:test` | Chạy toàn bộ test, **bỏ qua** GPU tests. |
| Test Suite (GPU) | `./gradlew :tests:test -PincludeGPU=true` | Chạy toàn bộ test, **bao gồm** GPU tests. Cần CUDA. |
| Test cụ thể | `./gradlew :tests:test --tests "com.user.nn.TestTensor"` | Chạy một test class cụ thể. |
| Clean & Re-test | `./gradlew :tests:cleanTest :tests:test` | Dọn cache và chạy lại mọi test. |
| CI Smoke | `powershell -File scripts\ci-test.ps1 -Mode quick` | Build + test + smoke examples. |

### Cấu hình Tags (JUnit 5)

Tags được cấu hình trong `tests/build.gradle.kts`. Gradle tự động bỏ qua GPU tests nếu không có cờ `-PincludeGPU=true`.

```groovy
test {
    useJUnitPlatform {
        if (!project.hasProperty('includeGPU')) {
            excludeTags 'gpu', 'gpu-smoke', 'gpu-nightly'
        }
    }
}
```

## 🤖 5. Quy trình CI/CD (Pipeline)

Hệ thống CI (GitHub Actions) thực hiện theo các giai đoạn:
1. **Giai đoạn 1 (Commit/PR)**: Chạy `./gradlew :tests:test`. Nếu fail, block PR ngay lập tức.
2. **Giai đoạn 2 (Nightly Build)**: Chạy `./gradlew :tests:test` bao gồm integration tests.
3. **Giai đoạn 3 (Hardware Runners)**: Chạy `./gradlew :tests:test -PincludeGPU=true` trên máy trạm chuyên dụng.
4. **Giai đoạn 4 (Performance)**: Chạy benchmark tasks để so sánh hiệu năng giữa các phiên bản.

## 🧠 6. Quy tắc cho Contributors (Best Practices)

✅ **Luôn Fix Seed**: Sử dụng `Torch.manual_seed(42)` trong mọi test case liên quan đến ngẫu nhiên.
✅ **Float Delta**: Luôn dùng `assertEquals(expected, actual, 1e-5)` cho float/Tensor.
✅ **Shape Assertions**: Luôn verify output shapes: `assertArrayEquals(expectedShape, result.shape)`.
✅ **Resource Management**: Dùng `MemoryScope` cho GPU training loops.
✅ **No `main()`**: Tất cả test phải là `@Test` methods, không dùng `public static void main`.
❌ **Không dùng `System.out` cho validation**: Sử dụng Assertions message.
❌ **Không dùng `System.exit`**: Để assertions fail tự nhiên.
❌ **Không phụ thuộc thứ tự**: Các Test không được phép phụ thuộc vào kết quả của test đứng trước.

## ⚠️ 7. Các Anti-patterns cần tránh

- **Silent Catch**: Tuyệt đối không `catch (Exception e) {}` mà không rethrow hoặc fail test.
- **GPU in CI core**: Không để code chạy GPU làm đỏ CI khi môi trường không có card.
- **Manual Validation**: Tránh dùng `if (result != expected) print(error)`. Hãy dùng `assertEquals`.
- **Shared Mutable State**: Không chia sẻ state giữa các test methods.

## 📊 8. Trạng thái hiện tại (2026-05-11)

| Metric | Giá trị |
|--------|---------|
| Test files | 50 |
| Test methods | 118+ |
| CPU tests | 100% PASSED |
| GPU tests | Excluded by default, pass with `-PincludeGPU=true` |
| `main()` runners | 0 (100% migrated) |
| Build system | Gradle 8.10+ Kotlin DSL |