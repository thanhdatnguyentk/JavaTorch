## ML_framework — Minimal Java neural-net library

This repository is a small re-implementation of a subset of `torch` in Java for learning and experimentation. It provides basic tensor-like utilities, a `Module`/`Parameter` system, several layers (linear, conv, pooling), activations, simple normalization and loss functions, and lightweight testing harnesses that compare outputs against NumPy reference implementations.

Current layout

- `src/`: library source code (core implementation in `com.user.nn.nn`)
- `bin/`: compiled classes (javac -d bin ...)
- `tests/`: testing assets
	- `tests/java/...`: Java test runners (unit-test-like programs)
	- `tests/*.py`: NumPy reference scripts used by tests

Progress (implementations added)

- Core: `Module`, `Parameter`, containers (`Sequential`, `ModuleList`, `ModuleDict`) — done
- Linear layer and activations: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softplus` — done
- Dropout, `BatchNorm1d` — done
- Loss functions: `mse_loss`, `cross_entropy_logits` — done
- Conv2d (naive), MaxPool2d, AvgPool2d, ZeroPad2d — done
- Matrix utilities, deterministic RNG and CSV IO helpers — done

Planned next work

- ConvTranspose / LazyConv / Conv1d/3d
- BatchNorm2d/3d, GroupNorm, LayerNorm, InstanceNorm
- RNNs (RNN/LSTM/GRU) and Transformer blocks
- More comprehensive tests and JUnit integration

Recent activity (latest)

- Added a basic `Tensor` class (`src/com/user/nn/Tensor.java`) and a `Torch` helper (`src/com/user/nn/Torch.java`) providing tensor creation, rand/randn, basic elementwise ops, matmul, and conversions to/from the existing `Mat` API.
- Refactored core `nn` matrix operations in `src/com/user/nn/nn.java` to delegate to the `Tensor`/`Torch` utilities while preserving the `Mat`-based public API.
- Compiled and ran the Java unit test suite; all Java tests passed:
	- `TestMatOps`, `TestContainers`, `TestParameterAndModules`, `TestFunctional`, `TestLinearReLU`, `TestActivations`, `TestLossesAndNorms`, `TestConvPool`.

Next recommended steps

- Continue migrating public APIs from `Mat` to `Tensor` (incrementally replace inputs/outputs in layers and examples).
- Implement prioritized Tensor ops: indexing/slicing, broadcasting arithmetic, reductions, and additional linear algebra (e.g., `svd`, `solve`) as needed.
- Add JUnit-based test harness and a `run-tests` script to automate builds and test runs.


Running tests (quick)

1. Build the library and tests:

```bash
javac -d bin src/com/user/nn/*.java
javac -d bin tests/java/com/user/nn/*.java
```

2. Run individual Java tests (examples):

```bash
java -cp bin com.user.nn.TestMatOps
java -cp bin com.user.nn.TestLinearReLU   # runs Python reference (requires Python3 numpy)
java -cp bin com.user.nn.TestConvPool     # runs conv Python reference
```

3. Python reference scripts require NumPy (Python 3.10 recommended):

```bash
python -m pip install numpy
python tests/linear_relu_ref.py tests/tmp/input.csv tests/tmp/weight.csv tests/tmp/bias.csv tests/tmp/out.csv
```


