# ML_framework — TODO

Last updated: 2026-03-05

## Current progress (completed)
- Core Module/Parameter system and containers
- `Tensor` class (shape, data, reshape/view, indexing, inplace ops)
- `Torch` helpers: creation ops, rand/randn/randint, bernoulli, multinomial
- Broadcasting-aware elementwise ops and binary helpers
- Shape ops: `cat`, `stack`, `split`, `chunk`, `permute`, `reshape`, `squeeze`, `unsqueeze`, `flatten`
- Math/trig/log functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`, `log10`, `log2`, `ceil`, `floor`, `round`, `trunc`
- Comparisons and `where`
- Reductions: `sum`, `mean`, `prod`, `max`, `min`, `argmax`, `argmin`, `var`, `std`, `norm`
- Linear algebra: `matmul`/`mm`/`bmm`, `dot`, `inverse`, `det`
- IO: `save` / `load`
- Grad-mode stubs: `no_grad` / `enable_grad` / `is_grad_enabled`
- Comprehensive Java test runners and automated `tests/run-tests.ps1` (all tests passing)

## Next-priority tasks
1. Expand Autograd Support
   - `backward` for basic shape operations (`reshape`, `view`, `squeeze`, `unsqueeze`, `flatten`, `permute`)
   - `backward` for reductions (`sum`, `mean`)
   - `backward` for matrix ops (`matmul`)
   - Explicitly add unit tests in `TestAutogradSimple` (or new classes) per method. Add them to `run-tests.ps1`.
2. Migrate `nn` framework to `Tensor` API
   - `Parameter` class to fully wrap `Tensor`
   - `Linear`, `ReLU`, `Sigmoid`, etc. to act natively on `Tensor`
   - `Sequential` and `Module` to pass `Tensor` between layers
3. Advanced indexing & broadcasting improvements
   - Full `gather`/`scatter` semantics, `take`, `index_select`
4. Full linear algebra suite
   - `solve`, `cholesky`, `svd`, `eigen` (use native Java libs if needed)
5. Testing & CI
   - Write integration test: Tensor-based MLP with end-to-end backprop
   - Add a GitHub Actions workflow to run `javac` + tests

## Immediate recommended step
- Start Phase 2: Expand Autograd Support by setting up hooks for shape ops, reduction ops, and matrix operations, strictly paired with tests.

---
If you confirm, I'll begin Phase 2: Expanding autograd.
