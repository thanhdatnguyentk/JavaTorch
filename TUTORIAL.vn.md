# Tutorial: Bắt đầu với ML Framework

[English](TUTORIAL.md) | [README](README.vn.md) | [API Reference](API_REFERENCE.vn.md) | [API Reference EN](API_REFERENCE.md)

Tài liệu này hướng dẫn theo lộ trình thực hành, dành cho người muốn hiểu và dùng framework ngay trên codebase hiện tại.

## 1. Mục tiêu của tutorial

Sau khi đi hết tài liệu này, bạn sẽ làm được 4 việc:

1. Build và chạy toàn bộ framework bằng Gradle với Java 21.
2. Chạy test để xác nhận môi trường hoạt động đúng.
3. Chạy các ví dụ huấn luyện có sẵn.
4. Tự viết một model nhỏ với `Tensor`, `Sequential`, autograd và optimizer.

## 2. Chuẩn bị môi trường

### Bắt buộc

- JDK 21+
- Gradle 8+ (chỉ cần trước khi đã có wrapper)
- thư mục làm việc đang đứng ở root repo

### Kiểm tra nhanh

```powershell
java -version
```

Bạn nên thấy Java 21 hoạt động bình thường.

## 3. Build project

### Cách chuẩn nhất (Gradle)

```powershell
gradle wrapper
.\gradlew.bat :core:clean :core:test :core:build
```

Trên macOS/Linux:

```bash
./gradlew :core:clean :core:test :core:build
```

Lý do nên dùng cách này trước:

- nó compile toàn bộ source
- compile và chạy test theo module
- giúp bạn biết ngay môi trường có thiếu dependency hay không

### Đường legacy (tương thích script cũ)

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
```

## 4. Chạy ví dụ đầu tiên: Iris

Đây là điểm bắt đầu tốt nhất vì:

- dữ liệu nhỏ
- code dễ đọc
- có đủ data loading, preprocessing, model, training loop, evaluation

### Chạy lệnh

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainIris
```

### Điều gì sẽ xảy ra

- framework tải file Iris nếu chưa có
- chuẩn hóa dữ liệu
- tạo `Sequential` model
- dùng `Adam`
- chuyển model lên GPU nếu path GPU khả dụng
- train nhiều epoch bằng `DataLoader`

### Bạn nên đọc file nào

- `src/com/user/nn/examples/TrainIris.java`
- `src/com/user/nn/containers/Sequential.java`
- `src/com/user/nn/layers/Linear.java`
- `src/com/user/nn/core/Tensor.java`
- `src/com/user/nn/core/Torch.java`

## 5. Chạy ví dụ dùng mini-batch thực tế: Fashion-MNIST

Ví dụ này cho thấy cách framework dùng dataloader và batching trên dữ liệu lớn hơn.

### Chạy lệnh

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainFashionMNIST
```

### Bạn sẽ học được gì

- cách tạo `Dataset`
- cách `DataLoader` sinh batch
- cách dùng `MemoryScope` trong training loop
- cách tách label CPU và input GPU
- cách evaluate model sau mỗi epoch

## 6. Chạy ví dụ NLP: Sentiment Analysis

Ví dụ này kết nối toàn bộ NLP stack trong repo.

### Chạy lệnh

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainSentiment
```

### Các thành phần chính

- `MovieCommentLoader`
- `Vocabulary`
- `BasicTokenizer`
- `Embedding`
- `LSTM`
- `SentimentModel`

Nếu bạn muốn hiểu `Embedding` và backward path trên GPU, đây là ví dụ quan trọng nên đọc.

## 7. Chạy ví dụ Transformer/Vision Transformer

### Vision Transformer trên CIFAR-10

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainViTCifar10
```

Ví dụ này minh họa:

- patch embedding
- attention
- encoder block
- scheduler
- evaluate theo epoch
- lưu model sau train

## 8. Mô hình lập trình cốt lõi của framework

Framework này xoay quanh 5 khối chính.

### Tensor

`Tensor` chứa dữ liệu, shape, gradient, trạng thái thiết bị và graph metadata.

Ví dụ tạo tensor:

```java
Tensor x = Torch.tensor(new float[] {1f, 2f, 3f, 4f}, 2, 2);
Tensor y = Torch.tensor(new float[] {5f, 6f, 7f, 8f}, 2, 2);
Tensor z = Torch.matmul(x, y);
```

### Autograd

Muốn tensor tham gia backward, bật:

```java
x.requires_grad = true;
```

Sau đó tính loss và gọi:

```java
loss.backward();
```

Framework sẽ xây topo order và lan truyền gradient từ tensor gốc về dependency.

### Module

Layer và model được tổ chức theo `Module` tương tự PyTorch. Bạn có thể gom layer bằng `Sequential`.

```java
Sequential model = new Sequential();
model.add(new Linear(4, 16, true));
model.add(new ReLU());
model.add(new Linear(16, 3, true));
```

### Optimizer

```java
Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.001f);
```

### DataLoader

Framework có `Data.Dataset` và `Data.DataLoader` để bạn xây pipeline mini-batch mà không cần thư viện ngoài.

## 9. Viết model đầu tiên của bạn

Ví dụ dưới đây là một classifier nhỏ để bạn thấy toàn bộ loop làm việc.

```java
import com.user.nn.core.*;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import com.user.nn.optim.*;

public class MiniClassifier {
    public static void main(String[] args) {
        Sequential model = new Sequential();
        model.add(new Linear(4, 16, true));
        model.add(new ReLU());
        model.add(new Linear(16, 3, true));

        for (Parameter p : model.parameters()) {
            p.getTensor().requires_grad = true;
        }

        Optim.Adam optimizer = new Optim.Adam(model.parameters(), 0.01f);

        Tensor x = Torch.tensor(new float[] {
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f
        }, 2, 4);

        int[] y = new int[] {0, 2};

        for (int step = 0; step < 200; step++) {
            optimizer.zero_grad();
            Tensor logits = model.forward(x);
            Tensor loss = Functional.cross_entropy_tensor(logits, y);
            loss.backward();
            optimizer.step();

            if (step % 20 == 0) {
                System.out.println("step=" + step + " loss=" + loss.data[0]);
            }
        }
    }
}
```

### Giải thích loop

1. `zero_grad()` xóa gradient cũ.
2. `forward()` tạo output và graph.
3. `cross_entropy_tensor()` tạo loss tensor.
4. `backward()` chạy autograd.
5. `step()` cập nhật tham số.

## 10. Dùng GPU đúng cách

### Chuyển model và tensor lên GPU

```java
model.toGPU();
x.toGPU();
```

Hoặc với tensor:

```java
x.to(Tensor.Device.GPU);
```

### Dùng `MemoryScope` trong loop

Đây là pattern nên ưu tiên khi train:

```java
for (Tensor[] batch : loader) {
    try (MemoryScope scope = new MemoryScope()) {
        Tensor xBatch = batch[0].to(Tensor.Device.GPU);
        optimizer.zero_grad();
        Tensor logits = model.forward(xBatch);
        Tensor loss = Functional.cross_entropy_tensor(logits, labels);
        loss.backward();
        optimizer.step();
    }
}
```

Mục tiêu của pattern này:

- giảm rò rỉ tensor tạm
- giảm chi phí cấp phát VRAM
- giữ loop ổn định lâu dài

## 11. Weight initialization

Repo hiện đã có API kiểu PyTorch trong `Torch.nn.init`.

Ví dụ:

```java
Torch.nn.init.zeros_(tensor);
Torch.nn.init.ones_(tensor);
Torch.nn.init.uniform_(tensor, -0.1f, 0.1f);
Torch.nn.init.normal_(tensor, 0f, 0.02f);
Torch.nn.init.xavier_uniform_(tensor);
Torch.nn.init.kaiming_uniform_(tensor);
```

Điều này phù hợp khi bạn viết layer mới hoặc muốn kiểm soát hội tụ tốt hơn so với random fill đơn giản.

## 12. In-place operations và lưu ý quan trọng

Framework hiện hỗ trợ các phép in-place như:

- `add_(float)`
- `sub_(float)`
- `mul_(float)`
- `add_(Tensor)`
- `sub_(Tensor)`
- `mul_(Tensor)`

Nhưng cần nhớ:

- in-place có thể phá computation graph nếu bạn sửa tensor đã được dùng trong forward
- framework có version counter để phát hiện trường hợp này
- nếu backward báo tensor bị sửa in-place, đó thường là lỗi logic thật, không nên lờ đi

## 13. Chạy test khi bạn sửa code

### Toàn bộ suite

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
```

### Một test riêng lẻ

```powershell
javac --add-modules jdk.incubator.vector -d bin -cp "bin;lib/*" tests\java\com\user\nn\TestInPlaceOps.java
java  --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.TestInPlaceOps
```

### Các test bạn nên nhớ tên

- `TestAutogradSimple`
- `TestAutogradMatmul`
- `TestAutogradConv`
- `TestAutogradEmbedding`
- `TestGPUKernels`
- `TestGPUEmbedding`
- `TestWeightInit`
- `TestInPlaceOps`
- `TestBlasOps`

## 14. Nếu muốn thêm một op mới

Quy trình khuyến nghị:

1. Thêm logic CPU vào `Torch.java` hoặc `Tensor.java`.
2. Nếu op cần gradient, nối vào `GradFn` phù hợp.
3. Nếu muốn tăng tốc GPU, thêm wrapper trong `CUDAOps.java`.
4. Nếu cần kernel riêng, thêm vào `src/com/user/nn/core/kernels.cu` rồi build lại PTX.
5. Viết test hồi quy.

## 15. Các lỗi thường gặp

### Lỗi `jdk.incubator.vector`

Nguyên nhân: quên `--add-modules jdk.incubator.vector` khi compile hoặc run.

### Lỗi classpath

Nguyên nhân: quên `-cp "bin;lib/*"` trên Windows.

### Lỗi GPU không khả dụng

Nguyên nhân có thể là:

- thiếu driver CUDA runtime
- thiếu native JAR phù hợp
- môi trường không có GPU NVIDIA

Framework thường sẽ fallback một phần về CPU, nhưng bạn vẫn nên xem log để xác nhận đường đi thực tế.

### Lỗi do in-place mutation

Nguyên nhân: bạn sửa tensor sau khi tensor đó đã được dùng để tạo graph forward.

Cách xử lý:

- tránh dùng in-place trong đoạn đang cần gradient nếu chưa chắc logic
- hoặc clone tensor trước khi sửa

## 16. Lộ trình học repo này hiệu quả nhất

1. `TrainIris`
2. `TrainFashionMNIST`
3. `TestAutogradSimple`
4. `TestAutogradMatmul`
5. `TrainSentiment`
6. `TrainViTCifar10`
7. `CUDAOps.java` và `kernels.cu`

Nếu đi theo thứ tự này, bạn sẽ hiểu từ API mức cao xuống kernel mức thấp mà không bị ngợp quá sớm.

## 17. Kết luận

Repo này phù hợp cho 3 kiểu mục tiêu:

- học cách tự xây một mini deep learning framework
- nghiên cứu và thử nghiệm thuật toán ngay trong Java
- mở rộng một codebase sẵn có theo hướng hiệu năng CPU/GPU

Khi cần bắt đầu nhanh, hãy quay lại đúng 2 lệnh này:

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainIris
```