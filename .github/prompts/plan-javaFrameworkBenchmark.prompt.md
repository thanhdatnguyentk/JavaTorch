## Plan: Benchmark ML Framework vs DL4J and PyTorch

Xây bộ benchmark end-to-end để so sánh framework của bạn với Deeplearning4j và PyTorch (Python) theo hai trục: độ chính xác hội tụ và hiệu suất huấn luyện/suy luận, trên cả CPU và GPU. Cách làm ưu tiên công bằng thực nghiệm: đồng nhất model/dataset/hyperparameter/seed, chuẩn hóa quy trình warmup + logging, và xuất artifact CSV để tổng hợp báo cáo tự động.

**Steps**
1. Phase 1 - Define Benchmark Contract (khóa chuẩn so sánh)
1. Chuẩn hóa benchmark matrix cho 2 task đã chọn: Classification (CIFAR-10 + ResNet18) và Sentiment (RT-Polarity + LSTM), với 2 thiết bị CPU/GPU và 3 framework (Your framework, DL4J, PyTorch Python).
1. Đặc tả bộ metric bắt buộc: final accuracy, best accuracy, epoch-to-target-accuracy, epoch time, total train time, inference latency (p50/p95), throughput (samples/s), memory footprint (heap và VRAM nếu có).
1. Khóa điều kiện fairness: seed cố định, batch size cố định theo task, số epoch cố định, optimizer/LR schedule giống nhau, preprocessing giống nhau, mixed precision mặc định tắt cho baseline tái lập.
1. Phase 2 - Build Benchmark Harness in This Repository
1. Tạo benchmark runner thống nhất cho framework của bạn, tái sử dụng loop từ TrainResNetCifar10 và TrainSentiment, đồng thời bổ sung timer chính xác cao, warmup nhất quán, và chuẩn output CSV/JSON. depends on 1
1. Bổ sung inference benchmark runner độc lập (model đã load trọng số) để đo latency/throughput tách khỏi training. depends on 5
1. Chuẩn hóa schema artifact: run_id, framework, task, device, seed, batch_size, epoch, train_loss, val_acc, epoch_time_ms, total_time_ms, throughput_sps, p50_latency_ms, p95_latency_ms, peak_heap_mb, peak_vram_mb. parallel with 6
1. Phase 3 - Add External Baselines (DL4J + PyTorch)
1. Tạo baseline script DL4J cho 2 task với cùng kiến trúc/hyperparameter và cùng output schema artifact. depends on 1
1. Tạo baseline script PyTorch Python cho 2 task với cùng contract và cùng output schema artifact để nhập chung pipeline báo cáo. depends on 1
1. Viết hướng dẫn map kiến trúc giữa 3 framework (ResNet18 block config, LSTM hidden size/layers/dropout, tokenizer/text pipeline) để tránh mismatch gây sai lệch accuracy. depends on 8 and 9
1. Phase 4 - Orchestrate and Aggregate Results
1. Tạo script orchestration chạy benchmark theo matrix (framework x task x device), có chế độ quick run (ít epoch) và full run (đủ epoch). depends on 7 and 8 and 9
1. Tạo aggregator đọc artifact và sinh bảng so sánh + biểu đồ (accuracy curve, throughput bar, latency distribution) + ranking theo từng tiêu chí. depends on 11
1. Định nghĩa scoring rule tổng hợp (ví dụ weighted score): 50% accuracy, 30% train throughput, 20% inference latency để có kết luận một con số cho từng task/device. depends on 12
1. Phase 5 - Reproducibility and Quality Gate
1. Viết checklist reproducibility: version Java/Python/CUDA/cuDNN, CPU/GPU model, driver, commit hash, dataset checksum, command line used. parallel with 12
1. Thêm sanity checks: phát hiện run lỗi (NaN loss, thiếu epoch, throughput âm, lệch seed), reject artifact không hợp lệ trước khi aggregate. depends on 12
1. Tạo báo cáo cuối dạng markdown cho mỗi lần benchmark: tóm tắt kết quả, phân tích trade-off accuracy vs performance, và kết luận theo từng task. depends on 13 and 15

**Relevant files**
- [src/com/user/nn/examples/TrainResNetCifar10.java](src/com/user/nn/examples/TrainResNetCifar10.java) - Mẫu training loop classification để tái sử dụng logic train/eval/history.
- [src/com/user/nn/examples/TrainSentiment.java](src/com/user/nn/examples/TrainSentiment.java) - Mẫu training NLP cho task sentiment.
- [src/com/user/nn/utils/visualization/TrainingHistory.java](src/com/user/nn/utils/visualization/TrainingHistory.java) - Cơ chế ghi metric theo epoch và export CSV hiện có, nên mở rộng schema.
- [src/com/user/nn/metrics/Accuracy.java](src/com/user/nn/metrics/Accuracy.java) - Metric accuracy chuẩn hóa giữa các task classification/sentiment.
- [src/com/user/nn/metrics/MetricTracker.java](src/com/user/nn/metrics/MetricTracker.java) - Điểm tích hợp thêm metric runtime và tổng hợp theo epoch.
- [src/com/user/nn/utils/progress/ProgressDataLoader.java](src/com/user/nn/utils/progress/ProgressDataLoader.java) - Nguồn throughput runtime để đối chiếu với timer benchmark.
- [src/com/user/nn/core/GpuMemoryPool.java](src/com/user/nn/core/GpuMemoryPool.java) - Điểm lấy telemetry VRAM cho benchmark GPU.
- [core/src/test/java/com/user/nn/TestGPUBenchmark.java](core/src/test/java/com/user/nn/TestGPUBenchmark.java) - Pattern warmup + CPU/GPU timing có thể tái cấu trúc thành harness chuẩn.
- [core/src/test/java/com/user/nn/TestBlasOps.java](core/src/test/java/com/user/nn/TestBlasOps.java) - Mẫu benchmark low-level, hữu ích cho micro baseline.
- [examples/trained_models/faster_rcnn_coco_history.csv](examples/trained_models/faster_rcnn_coco_history.csv) - Tham chiếu định dạng artifact lịch sử huấn luyện hiện tại.
- [README.md](README.md) - Cập nhật mục benchmark usage và cách chạy full matrix.

**Verification**
1. Chạy quick benchmark matrix (2 task x 3 framework x CPU) với epoch nhỏ để xác minh pipeline artifact hợp lệ và có thể aggregate.
1. Chạy full benchmark matrix trên CPU và GPU, lặp lại mỗi cấu hình ít nhất 3 lần để lấy median và độ lệch chuẩn.
1. Kiểm tra fairness checklist tự động: cùng seed, batch size, epoch, optimizer, LR schedule, preprocessing.
1. So khớp schema artifact giữa 3 framework, đảm bảo không thiếu cột bắt buộc và cùng đơn vị đo.
1. Đối chiếu accuracy curve để phát hiện mismatch kiến trúc/preprocessing (nếu chênh lệch bất thường ngay từ epoch đầu).
1. Xuất báo cáo markdown tổng hợp và xác nhận có đủ bảng: accuracy, throughput, latency, memory, weighted score.

**Decisions**
- In scope: benchmark end-to-end training và inference cho 2 task đã chọn (CIFAR-10/ResNet18, RT-Polarity/LSTM).
- In scope: baseline PyTorch Python chuẩn cộng với DL4J.
- In scope: chạy cả CPU và GPU.
- Out of scope hiện tại: object detection (COCO) vì chi phí thời gian/tài nguyên cao và khó khóa fairness nhanh trong vòng benchmark đầu.
- Out of scope hiện tại: mixed precision benchmark chính thức; chỉ mở sau khi baseline deterministic đã ổn định.

**Further Considerations**
1. Nên thêm một baseline CPU-only deterministic report làm “golden run” để regression về sau so sánh nhanh.
2. Nếu GPU VRAM hạn chế, ưu tiên giữ batch size cố định và giảm input pipeline overhead trước khi giảm model size.
3. Sau vòng 1 ổn định, có thể mở rộng thêm object detection như phase mở rộng độc lập để không làm nhiễu kết quả core benchmark.
