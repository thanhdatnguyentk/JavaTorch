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

- Migrate public APIs from `Mat` to `Tensor`: `nn.Module` layer components should purely use `Tensor`.
- Expand autograd: Add `.backward()` capability for reductions, linear algebra (matmul), shape modifications, and activations.
- Build a full `TestAutogradMLP.java` to test end-to-end backpropagation.
- Continue to strictly enforce unit tests per new feature added in autograd.


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



---

**Implemented Classes (status)**

- `Module`: implemented — base class for layers and containers.
- `Parameter`: implemented — simple wrapper for weight/bias `Mat`.
- `Sequential`: implemented — ordered container applying child modules.
- `ModuleList`: implemented — list-style container.
- `ModuleDict`: implemented — name->module mapping.
- `Linear`: implemented — dense layer (forward implemented, bias optional).
- `ReLU`: implemented — activation.
- `Sigmoid`: implemented — activation.
- `Tanh`: implemented — activation.
- `LeakyReLU`: implemented — activation.
- `Softplus`: implemented — activation.
- `Dropout`: implemented — stateless mask generation (seeded).
- `BatchNorm1d`: implemented — running mean/var and affine option.
- `Conv2d`: implemented — naive im2col per-sample implementation + convenience constructors.
- `MaxPool2d`: implemented — naive pooling.
- `AvgPool2d`: implemented — naive pooling.
- `ZeroPad2d`: implemented — padding utility.
- `Tensor`: implemented — basic tensor container (`shape`, `data`, `reshape/view`, indexing, scalar ops, inplace ops).
- `Torch`: implemented — helper utilities (tensor creation, rand/randn, matmul, elementwise ops, simple broadcasting, reductions, conversions to/from `Mat`).

Implemented `Torch` / `Tensor` functions (selected list)

- Tensor creation: `tensor` (overloads), `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `logspace`, `eye`
- Random: `rand`, `randn`, `randint`, `bernoulli`, `multinomial`, `manual_seed`
- Shape / indexing: `reshape`/`view`, `squeeze`, `unsqueeze`, `flatten`, `cat`, `stack`, `split`, `chunk`, `permute`
- Elementwise & math: `add`, `sub`, `mul`, `div`, `pow`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`, `log10`, `log2`, `ceil`, `floor`, `round`, `trunc`
- Comparisons: `eq`, `ne`, `gt`, `lt`, `ge`, `le`, `where`
- Broadcasting-aware ops: `binaryOp` / public wrappers (`add`,`mul`,...)
- Reductions: `sum`, `mean`, `prod`, `max`, `min`, `argmax`, `argmin`, `var`, `std`, `norm`
- Linear algebra: `matmul`/`mm`/`bmm`, `dot`, `inverse`, `det`
- IO & utils: `save`, `load`, `is_tensor`, `is_floating_point`, `set_default_dtype`, `get_default_dtype`, `set_printoptions`, `no_grad`/`enable_grad`/`is_grad_enabled`

Note: Many functions are basic/naive implementations (focus on clarity and correctness over performance). Advanced features (full advanced indexing, full linalg suite, complex dtypes, and autograd) remain TODO.

Recent updates (autograd + tests)

- Added a minimal autograd scaffold in `Tensor` supporting `requires_grad`, `grad`, `grad_fn`, and `backward()`.
- Implemented backward for selected ops: elementwise `mul`, `matmul`, and an autograd-aware scalar `sumTensor`.
- Added unit tests: `TestTorchExtras`, `TestTensor`, `TestGatherScatterExtras`, and `TestAutogradSimple` under `tests/java/com/user/nn/` and integrated them into `tests/run-tests.ps1`.
- Created `todo.md` at project root summarizing next-priority work (autograd expansion, advanced indexing, linalg, CI).
- All Java tests (including autograd simple tests) pass locally via `tests/run-tests.ps1`.

**Tests & Examples**

- Java test runners in `tests/java/com/user/nn/` — implemented and kept (MatOps, Containers, Functional, Linear+ReLU, Activations, LossesAndNorms, ConvPool).
- Python NumPy reference scripts in `tests/` — implemented (`linear_relu_ref.py`, `conv_ref.py`) and used by Java tests.
- `TrainIris.java` example — implemented and runnable (trained small network, reported final accuracy).

**Continuous README updates**

Last updated: 2026-03-05.

This repository's `README.md` will be updated continuously as work progresses. The "Implemented Classes" list above is the canonical, incrementally maintained status report. I'll update this section (and the top-level progress summary) whenever I implement or change classes, utilities, tests, or examples.


