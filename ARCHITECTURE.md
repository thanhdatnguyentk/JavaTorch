JavaTorch Testing & Automation Architecture

Tài liệu này mô tả kiến trúc hệ thống kiểm thử và quy trình tự động hóa của dự án JavaTorch. Mục tiêu là đảm bảo hệ thống nhanh (Fast CI), sâu (GPU validation) và đáng tin cậy (Benchmarks).

1. Nguyên tắc cốt lõi (Core Principles)
- Tách biệt mục đích: Unit test chạy mọi lúc; GPU test và Benchmark chỉ chạy khi có môi trường phù hợp.
- Độc lập với phần cứng: Code core (Tensor logic) phải chạy được trên CPU để CI luôn xanh.
- Mọi thứ qua Gradle: Toàn bộ vòng đời từ build, test đến benchmark đều được quản lý tập trung.
- Tính nhất quán (Determinism): Mọi test case liên quan đến Random/Weights phải được fix seed.

2. Cấu trúc thư mục (Project Structure)

Dự án được phân cấp để tách biệt mã nguồn chính và các loại hình kiểm thử khác nhau:

PlaintextJavaTorch/
├── core/
│   ├── src/main/java/            # Mã nguồn chính (Tensor, Neural Networks, Ops)
│   └── src/test/java/            # Unit Tests cơ bản cho module core
│
├── tests/
│   ├── java/com/user/nn/         # 100% JUnit 5 Tests (Unit, Integration, GPU)
│   │   ├── ...Test.java          # Các test logic tensor, autograd
│   │   ├── ...GpuSmokeTest.java  # GPU/CUDA Tests (cần có phần cứng)
│   │   └── ...IT.java            # Integration Tests
│   └── build.gradle.kts          # Cấu hình module tests
│
├── examples/                     # Các ví dụ huấn luyện (Fashion-MNIST, NLP)
├── build.gradle.kts              # Cấu hình Gradle Root (multi-module)
└── ARCHITECTURE.md               # Tài liệu kiến trúc (File này)

🧪 3. Chiến lược kiểm thử 4 tầng (4-Tier Strategy)
 3.1. Unit Tests (Mặc định - Fast CI)
  - Mục tiêu: Kiểm tra các phép toán Tensor đơn lẻ, tính toán đạo hàm, khởi tạo layer.
  - Ràng buộc: Chạy cực nhanh (<1s/test), không yêu cầu GPU.
  - Yêu cầu: Sai số số thực (Delta) tối đa 1e-6.
 3.2. Integration Tests
  - Mục tiêu: Kiểm tra luồng dữ liệu xuyên suốt (Model Training loop, Forward/Backward pass).
  - Tags: @Tag("integration").
 3.3. GPU Tests
  - Mục tiêu: Kiểm tra CUDA kernels và tính đúng đắn trên phần cứng tăng tốc.
  - Cơ chế: Sử dụng Assumptions.assumeTrue(GpuContext.isAvailable()) để tránh fail khi máy không có card đồ họa.
  - Tags: @Tag("gpu").
 3.4. Benchmarks
  - Mục tiêu: Đo lường hiệu năng (Ops/sec, Memory usage).
  - Lưu ý: Không chạy bằng JUnit. Sử dụng class main() hoặc thư viện JMH.

⚙️ 4. Cấu hình Gradle (Execution Control)

Các Task thực thi chính

| Task | Lệnh thực thi | Mô tả |
| :--- | :--- | :--- |
| Unit Test (Core) | `./gradlew :core:test` | Kiểm tra logic core của thư viện. Bỏ qua GPU tests. |
| Test Suite (Tất cả) | `./gradlew :tests:test` | Chạy toàn bộ test bao gồm Unit & Integration. |
| GPU Test Suite | `./gradlew :tests:test -PincludeGPU=true` | Chạy toàn bộ test, **bao gồm** các test yêu cầu GPU. Cần CUDA. |
| Clean Build | `./gradlew cleanTest test` | Dọn dẹp cache và chạy lại mọi bài kiểm tra. |

Cấu hình Tags (JUnit 5)
Trong cấu trúc mới, chúng ta không dùng SourceSets riêng biệt mà dùng `@Tag("gpu")` và `@Tag("slow")` để lọc các test. Gradle sẽ tự động bỏ qua GPU tests nếu không có cờ `-PincludeGPU=true`.


🤖 5. Quy trình CI/CD (Pipeline)
Hệ thống CI (GitHub Actions) sẽ thực hiện theo các giai đoạn:
Giai đoạn 1 (Commit/PR): Chạy ./gradlew test. Nếu fail, block PR ngay lập tức.
Giai đoạn 2 (Nightly Build): Chạy ./gradlew integrationTest.
Giai đoạn 3 (Hardware Runners): Chạy ./gradlew gpuTest trên các máy trạm chuyên dụng.
Giai đoạn 4 (Performance): Chạy ./gradlew benchmark để so sánh hiệu năng giữa các phiên bản.
🧠 6. Quy tắc cho Contributors (Best Practices)
✅ Luôn Fix Seed: Sử dụng RandomConfig.setSeed(42) trong mọi test case liên quan đến ngẫu nhiên.
✅ Resource Management: Tensors phải được giải phóng qua try-with-resources hoặc tensor.close().
✅ Timeout: Luôn đặt @Timeout(seconds = 5) cho các test case để tránh treo pipeline.
❌ Không dùng System.out: Sử dụng Assertions message hoặc Log4j để ghi lại thông tin lỗi.
❌ Không phụ thuộc thứ tự: Các Test không được phép phụ thuộc vào kết quả của test đứng trước nó.
⚠️ 7. Các Anti-patterns cần tránh
Silent Catch: Tuyệt đối không catch (Exception e) {} mà không rethrow hoặc fail test.
GPU in CI core: Không để code chạy GPU làm đỏ CI khi môi trường không có card.
Manual Validation: Tránh dùng if (result != expected) print(error). Hãy dùng assertEquals.