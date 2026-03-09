# Tutorial: Getting Started with ML Framework

[Tiếng Việt](TUTORIAL.md) | [README](README.en.md) | [API Reference](API_REFERENCE.en.md) | [API Reference VI](API_REFERENCE.md)

This tutorial is a practical onboarding path for the current codebase.

## 1. What you will achieve

By the end of this tutorial, you should be able to:

1. Build and run the full framework on Windows with Java 21.
2. Run the regression suite to validate your environment.
3. Execute the provided training examples.
4. Write a small model using `Tensor`, `Sequential`, autograd, and an optimizer.

## 2. Environment setup

### Required

- Windows + PowerShell
- JDK 21+
- current working directory at the repository root

### Quick check

```powershell
java -version
javac -version
```

## 3. Build the project

### Recommended path

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
```

This script compiles all source files, compiles all tests, and runs the full regression suite.

### If you only want to compile

```powershell
javac --add-modules jdk.incubator.vector -d bin -cp "lib/*" `
    src\com\user\nn\core\*.java `
    src\com\user\nn\layers\*.java `
    src\com\user\nn\activations\*.java `
    src\com\user\nn\containers\*.java `
    src\com\user\nn\norm\*.java `
    src\com\user\nn\pooling\*.java `
    src\com\user\nn\rnn\*.java `
    src\com\user\nn\attention\*.java `
    src\com\user\nn\losses\*.java `
    src\com\user\nn\optim\*.java `
    src\com\user\nn\dataloaders\*.java `
    src\com\user\nn\models\*.java `
    src\com\user\nn\models\cv\*.java `
    src\com\user\nn\models\generative\*.java `
    src\com\user\nn\metrics\*.java `
    src\com\user\nn\examples\*.java
```

## 4. First example: Iris

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainIris
```

This is the best entry point because it is small, readable, and complete.

## 5. Fashion-MNIST example

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainFashionMNIST
```

This example shows realistic mini-batch training with `Dataset`, `DataLoader`, `MemoryScope`, evaluation, and GPU-aware execution.

## 6. Sentiment analysis example

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainSentiment
```

This is the main NLP walkthrough in the repository and uses `Vocabulary`, `BasicTokenizer`, `Embedding`, and `LSTM`.

## 7. Vision Transformer example

```powershell
java --add-modules jdk.incubator.vector -cp "bin;lib/*" com.user.nn.examples.TrainViTCifar10
```

This example demonstrates patch embedding, attention, encoder blocks, scheduling, evaluation, and model saving.

## 8. Core programming model

### Tensor

```java
Tensor x = Torch.tensor(new float[] {1f, 2f, 3f, 4f}, 2, 2);
Tensor y = Torch.tensor(new float[] {5f, 6f, 7f, 8f}, 2, 2);
Tensor z = Torch.matmul(x, y);
```

### Autograd

```java
x.requires_grad = true;
loss.backward();
```

### Module composition

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

## 9. Your first model

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
        }
    }
}
```

## 10. GPU usage

```java
model.toGPU();
x.to(Tensor.Device.GPU);
```

Recommended loop pattern:

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

## 11. Weight initialization

```java
Torch.nn.init.zeros_(tensor);
Torch.nn.init.ones_(tensor);
Torch.nn.init.uniform_(tensor, -0.1f, 0.1f);
Torch.nn.init.normal_(tensor, 0f, 0.02f);
Torch.nn.init.xavier_uniform_(tensor);
Torch.nn.init.kaiming_uniform_(tensor);
```

## 12. In-place operations

The framework supports:

- `add_(float)`
- `sub_(float)`
- `mul_(float)`
- `add_(Tensor)`
- `sub_(Tensor)`
- `mul_(Tensor)`

Version tracking is used to detect invalid graph mutations caused by in-place updates.
