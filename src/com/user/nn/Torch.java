package com.user.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class Torch {
    public static String defaultDtype = "float32";
    public static PrintOptions printOptions = new PrintOptions();
    private static Random globalR = new Random(0);

    public static class PrintOptions {
        public int precision = 6;
        public int threshold = 1000;
    }

    // dtype helpers
    public static boolean is_tensor(Object o) {
        return o instanceof Tensor;
    }

    public static boolean is_floating_point(Tensor t) {
        return t != null;
    }

    public static void set_default_dtype(String d) {
        defaultDtype = d;
    }

    public static String get_default_dtype() {
        return defaultDtype;
    }

    public static void set_printoptions(int precision, int threshold) {
        printOptions.precision = precision;
        printOptions.threshold = threshold;
    }

    public static void manual_seed(long s) {
        globalR = new Random(s);
    }

    // Creation
    public static Tensor tensor(float[] data, int... shape) {
        return new Tensor(data, shape);
    }

    public static Tensor zeros(int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = 0f;
        return t;
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = 1f;
        return t;
    }

    public static Tensor full(int[] shape, float value) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = value;
        return t;
    }

    public static Tensor arange(int start, int end) {
        int n = end - start;
        float[] d = new float[n];
        for (int i = 0; i < n; i++)
            d[i] = start + i;
        return new Tensor(d, n);
    }

    public static Tensor eye(int n) {
        Tensor t = zeros(n, n);
        for (int i = 0; i < n; i++)
            t.data[i * n + i] = 1f;
        return t;
    }

    public static Tensor rand(int[] shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = globalR.nextFloat();
        return t;
    }

    public static Tensor randn(int[] shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = (float) nextGaussian(globalR);
        return t;
    }

    // bernoulli sampling from probability tensor p (each entry is probability of 1)
    public static Tensor bernoulli(Tensor p) {
        Tensor out = new Tensor(p.shape);
        for (int i = 0; i < p.data.length; i++)
            out.data[i] = (globalR.nextDouble() < p.data[i]) ? 1f : 0f;
        return out;
    }

    public static Tensor bernoulli(float prob, int... shape) {
        Tensor p = full(shape, prob);
        return bernoulli(p);
    }

    // multinomial sampling: supports 1D probs (length N) or 2D batch (B x N)
    // returns indices as float values in a tensor of shape (num_samples,) or (B,
    // num_samples)
    public static Tensor multinomial(Tensor probs, int num_samples, boolean replacement) {
        if (probs.shape.length == 1) {
            int N = probs.shape[0];
            float[] work = probs.data.clone();
            float sum = 0f;
            for (float v : work)
                sum += v;
            if (sum <= 0f)
                throw new IllegalArgumentException("multinomial: probabilities must sum to >0");
            Tensor out = new Tensor(num_samples);
            for (int s = 0; s < num_samples; s++) {
                double r = globalR.nextDouble() * sum;
                double acc = 0.0;
                int chosen = -1;
                for (int i = 0; i < N; i++) {
                    acc += work[i];
                    if (r < acc) {
                        chosen = i;
                        break;
                    }
                }
                if (chosen < 0)
                    chosen = N - 1;
                out.data[s] = chosen;
                if (!replacement) {
                    sum -= work[chosen];
                    work[chosen] = 0f;
                    if (sum <= 0f && s < num_samples - 1)
                        throw new IllegalArgumentException("multinomial: not enough mass to draw without replacement");
                }
            }
            return out;
        } else if (probs.shape.length == 2) {
            int B = probs.shape[0], N = probs.shape[1];
            Tensor out = new Tensor(B, num_samples);
            for (int b = 0; b < B; b++) {
                float[] work = new float[N];
                System.arraycopy(probs.data, b * N, work, 0, N);
                float sum = 0f;
                for (float v : work)
                    sum += v;
                if (sum <= 0f)
                    throw new IllegalArgumentException("multinomial: probabilities must sum to >0");
                for (int s = 0; s < num_samples; s++) {
                    double r = globalR.nextDouble() * sum;
                    double acc = 0.0;
                    int chosen = -1;
                    for (int i = 0; i < N; i++) {
                        acc += work[i];
                        if (r < acc) {
                            chosen = i;
                            break;
                        }
                    }
                    if (chosen < 0)
                        chosen = N - 1;
                    out.data[b * num_samples + s] = chosen;
                    if (!replacement) {
                        sum -= work[chosen];
                        work[chosen] = 0f;
                        if (sum <= 0f && s < num_samples - 1)
                            throw new IllegalArgumentException(
                                    "multinomial: not enough mass to draw without replacement");
                    }
                }
            }
            return out;
        }
        throw new UnsupportedOperationException("multinomial: only 1D or 2D probs supported");
    }

    public static Tensor multinomial(Tensor probs, int num_samples) {
        return multinomial(probs, num_samples, false);
    }

    private static double nextGaussian(Random r) { // Box-Muller
        double u = r.nextDouble();
        double v = r.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    // Basic math (elementwise) - assumes same shape
    public static Tensor sub(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x - y);
    }

    public static Tensor mul(Tensor a, Tensor b) {
        Tensor out = binaryOp(a, b, (x, y) -> x * y);
        // autograd for mul: dOut/dA = B * outGrad ; dOut/dB = A * outGrad
        if (a.requires_grad || b.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor ga = binaryOp(outGrad, b, (x, y) -> x * y); // outGrad * b
                        ga = reduceSumToShape(ga, a.shape);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        Tensor gb = binaryOp(outGrad, a, (x, y) -> x * y);
                        gb = reduceSumToShape(gb, b.shape);
                        b.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    // attach autograd for add
    public static Tensor addWithGrad(Tensor a, Tensor b) {
        Tensor out = binaryOp(a, b, (x, y) -> x + y);
        if (a.requires_grad || b.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor ga = reduceSumToShape(outGrad, a.shape);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        Tensor gb = reduceSumToShape(outGrad, b.shape);
                        b.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    public static Tensor div(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x / y);
    }

    // scalar variants
    public static Tensor add(Tensor a, float scalar) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = a.data[i] + scalar;
        return out;
    }

    public static Tensor mul(Tensor a, float scalar) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = a.data[i] * scalar;
        return out;
    }

    // public add that picks autograd-aware impl
    public static Tensor add(Tensor a, Tensor b) {
        return addWithGrad(a, b);
    }

    // scalar add/mul autograd not implemented (user can use elementwise ops with
    // tensors)

    private static Tensor binaryOp(Tensor a, Tensor b, FloatBinaryOp op) {
        // full broadcasting support for shapes with compatible dims
        int[] ash = a.shape.clone();
        int[] bsh = b.shape.clone();
        int na = ash.length, nb = bsh.length;
        int nout = Math.max(na, nb);
        int[] a2 = new int[nout];
        int[] b2 = new int[nout];
        for (int i = 0; i < nout; i++) {
            int ia = i - (nout - na);
            a2[i] = ia >= 0 ? ash[ia] : 1;
            int ib = i - (nout - nb);
            b2[i] = ib >= 0 ? bsh[ib] : 1;
            if (a2[i] != b2[i] && a2[i] != 1 && b2[i] != 1)
                throw new IllegalArgumentException("shapes not broadcastable");
        }
        int[] outShape = new int[nout];
        int outNum = 1;
        for (int i = 0; i < nout; i++) {
            outShape[i] = Math.max(a2[i], b2[i]);
            outNum *= outShape[i];
        }
        int[] aStr = computeStrides(a2);
        int[] bStr = computeStrides(b2);
        int[] outStr = computeStrides(outShape);
        Tensor out = new Tensor(outShape);
        for (int idx = 0; idx < outNum; idx++) {
            int rem = idx;
            int offA = 0, offB = 0;
            for (int d = 0; d < nout; d++) {
                int coord = rem / outStr[d];
                rem = rem % outStr[d];
                int aDim = a2[d];
                int bDim = b2[d];
                int ca = (aDim == 1) ? 0 : coord;
                int cb = (bDim == 1) ? 0 : coord;
                offA += ca * aStr[d];
                offB += cb * bStr[d];
            }
            out.data[idx] = op.apply(a.data[offA], b.data[offB]);
        }
        return out;
    }

    // Reduce outGrad (of shape grad.shape) to targetShape by summing broadcasted
    // dims
    private static Tensor reduceSumToShape(Tensor grad, int[] targetShape) {
        // if shapes equal just return grad.clone()
        if (java.util.Arrays.equals(grad.shape, targetShape))
            return grad.clone();
        int gnd = grad.shape.length;
        int tnd = targetShape.length;
        int nout = Math.max(gnd, tnd);
        int[] g2 = new int[nout];
        int[] t2 = new int[nout];
        for (int i = 0; i < nout; i++) {
            int ig = i - (nout - gnd);
            g2[i] = ig >= 0 ? grad.shape[ig] : 1;
            int it = i - (nout - tnd);
            t2[i] = it >= 0 ? targetShape[it] : 1;
        }
        int[] outShape = targetShape.clone();
        Tensor out = new Tensor(outShape);
        int[] gStr = computeStrides(g2);
        int[] outStr = computeStrides(outShape);
        int total = 1;
        for (int s : grad.shape)
            total *= s;
        for (int idx = 0; idx < total; idx++) {
            int rem = idx;
            int[] coord = new int[nout];
            for (int d = 0; d < nout; d++) {
                coord[d] = rem / gStr[d];
                rem = rem % gStr[d];
            }
            int outOff = 0;
            for (int d = 0; d < nout; d++) {
                int c = (t2[d] == 1) ? 0 : coord[d];
                int targetDim = d - (nout - outShape.length);
                if (targetDim < 0) {
                    /* ignored */ }
                outOff += c * (d < outStr.length ? outStr[d - (nout - outShape.length)] : 0);
            }
            // compute out linear index properly by mapping coordinates to out shape
            // simpler implementation: compute out coordinate array
            int[] outCoord = new int[outShape.length];
            int oidx = 0;
            for (int d = 0; d < outShape.length; d++) {
                int srcDim = d + (nout - outShape.length);
                int c = coord[srcDim];
                if (outShape[d] == 1)
                    outCoord[d] = 0;
                else
                    outCoord[d] = c;
            }
            int outLinear = 0;
            int[] outStr2 = computeStrides(outShape);
            for (int d = 0; d < outShape.length; d++)
                outLinear += outCoord[d] * outStr2[d];
            out.data[outLinear] += grad.data[idx];
        }
        return out;
    }

    private static int[] computeStrides(int[] shape) {
        int n = shape.length;
        int[] s = new int[n];
        if (n == 0)
            return s;
        int prod = 1;
        for (int i = n - 1; i >= 0; i--) {
            s[i] = prod;
            prod *= shape[i];
        }
        return s;
    }

    private interface FloatBinaryOp {
        float apply(float x, float y);
    }

    // reductions
    public static float sum(Tensor a) {
        float s = 0f;
        for (float v : a.data)
            s += v;
        return s;
    }

    public static float mean(Tensor a) {
        return sum(a) / a.numel();
    }

    // autograd-aware mean returning a 1-element Tensor
    public static Tensor meanTensor(Tensor a) {
        float m = mean(a);
        Tensor out = new Tensor(new float[] { m }, 1);
        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    float scale = outGrad.data[0] / a.numel();
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++)
                        ga.data[i] = scale;
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    // autograd-aware sum returning a 1-element Tensor
    public static Tensor sumTensor(Tensor a) {
        float s = sum(a);
        Tensor out = new Tensor(new float[] { s }, 1);
        if (a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    // outGrad is scalar (shape [1])
                    float scale = outGrad.data[0];
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++)
                        ga.data[i] = scale;
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor zeros_like(Tensor a) {
        return zeros(a.shape);
    }

    public static Tensor ones_like(Tensor a) {
        return ones(a.shape);
    }

    // sum along axis (supports axis==-1 for total sum or axis for 2D tensors)
    public static Tensor sum(Tensor a, int axis) {
        if (axis < 0)
            return new Tensor(new float[] { sum(a) }, 1);
        if (a.shape.length == 2 && axis == 0) {
            int r = a.shape[0], c = a.shape[1];
            Tensor out = new Tensor(1, c);
            for (int j = 0; j < c; j++) {
                float s = 0f;
                for (int i = 0; i < r; i++)
                    s += a.data[i * c + j];
                out.data[j] = s;
            }
            return out;
        }
        if (a.shape.length == 2 && axis == 1) {
            int r = a.shape[0], c = a.shape[1];
            Tensor out = new Tensor(r, 1);
            for (int i = 0; i < r; i++) {
                float s = 0f;
                for (int j = 0; j < c; j++)
                    s += a.data[i * c + j];
                out.data[i] = s;
            }
            return out;
        }
        throw new UnsupportedOperationException("sum along axis not implemented for this shape");
    }

    public static Tensor mean(Tensor a, int axis) {
        Tensor s = sum(a, axis);
        if (axis < 0)
            return new Tensor(new float[] { mean(a) }, 1);
        if (a.shape.length == 2) {
            if (axis == 0) {
                int c = a.shape[1];
                for (int j = 0; j < c; j++)
                    s.data[j] /= a.shape[0];
                return s;
            } else {
                int r = a.shape[0];
                for (int i = 0; i < r; i++)
                    s.data[i] /= a.shape[1];
                return s;
            }
        }
        throw new UnsupportedOperationException("mean axis not implemented");
    }

    // reshape / transpose utilities (simple)
    public static Tensor reshape(Tensor a, int... newShape) {
        return a.reshape(newShape);
    }

    public static Tensor view(Tensor a, int... newShape) {
        return reshape(a, newShape);
    }

    public static Tensor transpose(Tensor a, int dim0, int dim1) {
        if (a.shape.length != 2)
            throw new IllegalArgumentException("transpose only supports 2D currently");
        int r = a.shape[0], c = a.shape[1];
        Tensor out = new Tensor(c, r);
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                out.data[j * r + i] = a.data[i * c + j];

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor gradOutput) {
                    a.backwardStep(transpose(gradOutput, dim0, dim1));
                }
            };
        }
        return out;
    }

    // convenience 2D transpose
    public static Tensor transpose(Tensor a) {
        return transpose(a, 0, 1);
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.shape.length != 2 || b.shape.length != 2)
            throw new IllegalArgumentException("matmul supports 2D tensors");
        int m = a.shape[0], k = a.shape[1], n = b.shape[1];
        if (k != b.shape[0])
            throw new IllegalArgumentException("matmul shape mismatch");
        Tensor out = new Tensor(m, n);

        // We optimize matmul by transposing B once, so memory access is contiguous
        Tensor bt = transpose(b);
        int upperBound = SPECIES.loopBound(k);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0f;
                FloatVector sumVector = FloatVector.zero(SPECIES);
                int kk = 0;

                // SIMD loop
                for (; kk < upperBound; kk += SPECIES.length()) {
                    FloatVector va = FloatVector.fromArray(SPECIES, a.data, i * k + kk);
                    FloatVector vb = FloatVector.fromArray(SPECIES, bt.data, j * k + kk);
                    sumVector = sumVector.add(va.mul(vb));
                }

                sum += sumVector.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);

                // Tail loop
                for (; kk < k; kk++) {
                    sum += a.data[i * k + kk] * bt.data[j * k + kk];
                }
                out.data[i * n + j] = sum;
            }
        }
        // autograd for matmul: dOut/dA = outGrad.matmul(B^T); dOut/dB =
        // A^T.matmul(outGrad)
        if (a.requires_grad || b.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor ga = matmul(outGrad, bt);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        Tensor at = transpose(a);
                        Tensor gb = matmul(at, outGrad);
                        b.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    // stack: insert a new dimension at `dim` and concatenate
    public static Tensor stack(List<Tensor> tensors, int dim) {
        if (tensors.size() == 0)
            throw new IllegalArgumentException("no tensors to stack");
        // unsqueeze each tensor at dim and cat
        ArrayList<Tensor> ups = new ArrayList<>();
        for (Tensor t : tensors)
            ups.add(t.unsqueeze(dim));
        return cat(ups, dim);
    }

    // split by sizes along dim
    public static List<Tensor> split(Tensor a, int[] sizes, int dim) {
        ArrayList<Tensor> out = new ArrayList<>();
        if (dim < 0)
            dim += a.shape.length;

        int offset = 0;
        for (int s : sizes) {
            out.add(narrow(a, dim, offset, s));
            offset += s;
        }
        return out;
    }

    // chunk into `chunks` parts along dim (last part may be smaller)
    public static List<Tensor> chunk(Tensor a, int chunks, int dim) {
        if (dim < 0)
            dim += a.shape.length;
        int total = a.shape[dim];
        int base = total / chunks;
        int rem = total % chunks;
        int[] sizes = new int[chunks];
        for (int i = 0; i < chunks; i++)
            sizes[i] = base + (i < rem ? 1 : 0);
        return split(a, sizes, dim);
    }

    // where: choose elements from x or y based on condition (cond != 0)
    public static Tensor where(Tensor cond, Tensor x, Tensor y) {
        if (cond.numel() != x.numel() || x.numel() != y.numel())
            throw new IllegalArgumentException("where: shapes must match elementwise");
        Tensor out = new Tensor(x.shape);
        for (int i = 0; i < out.data.length; i++)
            out.data[i] = (cond.data[i] != 0f) ? x.data[i] : y.data[i];
        return out;
    }

    // permute axes: dims is a permutation of [0..nd-1]
    public static Tensor permute(Tensor a, int... dims) {
        int nd = a.shape.length;
        if (dims.length != nd)
            throw new IllegalArgumentException("permute: dims length must match rank");
        boolean[] seen = new boolean[nd];
        for (int d : dims) {
            if (d < 0 || d >= nd || seen[d])
                throw new IllegalArgumentException("permute: invalid permutation");
            seen[d] = true;
        }
        int[] outShape = new int[nd];
        for (int i = 0; i < nd; i++)
            outShape[i] = a.shape[dims[i]];
        Tensor out = new Tensor(outShape);
        int[] inStr = computeStrides(a.shape);
        int[] outStr = computeStrides(outShape);
        int outNum = out.numel();
        for (int idx = 0; idx < outNum; idx++) {
            int rem = idx;
            int[] outCoord = new int[nd];
            for (int i = 0; i < nd; i++) {
                outCoord[i] = rem / outStr[i];
                rem = rem % outStr[i];
            }
            int[] inCoord = new int[nd];
            for (int i = 0; i < nd; i++)
                inCoord[dims[i]] = outCoord[i];
            int inOff = 0;
            for (int i = 0; i < nd; i++)
                inOff += inCoord[i] * inStr[i];
            out.data[idx] = a.data[inOff];
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor gradOutput) {
                    // Reverse permutation
                    int[] revDims = new int[nd];
                    for (int i = 0; i < nd; i++)
                        revDims[dims[i]] = i;
                    a.backwardStep(permute(gradOutput, revDims));
                }
            };
        }
        return out;
    }

    // gather: for each position, take input value at index specified along `dim`
    public static Tensor gather(Tensor input, int dim, Tensor index) {
        if (dim < 0)
            dim += input.shape.length;
        if (input.shape.length != index.shape.length)
            throw new IllegalArgumentException("gather: rank mismatch");
        for (int i = 0; i < input.shape.length; i++)
            if (i != dim && input.shape[i] != index.shape[i])
                throw new IllegalArgumentException("gather: shapes must match except at dim");
        int nd = input.shape.length;
        int[] inStr = computeStrides(input.shape);
        int[] idxStr = computeStrides(index.shape);
        Tensor out = new Tensor(index.shape);
        int ne = out.numel();
        for (int linear = 0; linear < ne; linear++) {
            int rem = linear;
            int[] coord = new int[nd];
            for (int i = 0; i < nd; i++) {
                coord[i] = rem / idxStr[i];
                rem = rem % idxStr[i];
            }
            int idxAt = (int) index.data[linear];
            // allow negative indices
            if (idxAt < 0)
                idxAt += input.shape[dim];
            if (idxAt < 0 || idxAt >= input.shape[dim])
                throw new IndexOutOfBoundsException("gather: index out of range");
            int inOff = 0;
            for (int i = 0; i < nd; i++) {
                int c = (i == dim) ? idxAt : coord[i];
                inOff += c * inStr[i];
            }
            out.data[linear] = input.data[inOff];
        }
        return out;
    }

    // scatter: write values from src into input at positions specified by index
    // along dim (returns a new tensor)
    public static Tensor scatter(Tensor input, int dim, Tensor index, Tensor src) {
        if (dim < 0)
            dim += input.shape.length;
        if (input.shape.length != index.shape.length || !java.util.Arrays.equals(index.shape, src.shape))
            throw new IllegalArgumentException("scatter: shapes mismatch");
        int nd = input.shape.length;
        int[] inStr = computeStrides(input.shape);
        int[] idxStr = computeStrides(index.shape);
        Tensor out = input.clone();
        int ne = index.numel();
        for (int linear = 0; linear < ne; linear++) {
            int rem = linear;
            int[] coord = new int[nd];
            for (int i = 0; i < nd; i++) {
                coord[i] = rem / idxStr[i];
                rem = rem % idxStr[i];
            }
            int idxAt = (int) index.data[linear];
            // allow negative indices
            if (idxAt < 0)
                idxAt += input.shape[dim];
            if (idxAt < 0 || idxAt >= input.shape[dim])
                throw new IndexOutOfBoundsException("scatter: index out of range");
            int inOff = 0;
            for (int i = 0; i < nd; i++) {
                int c = (i == dim) ? idxAt : coord[i];
                inOff += c * inStr[i];
            }
            out.data[inOff] = src.data[linear];
        }
        return out;
    }

    // in-place scatter: mutates `input` and returns it
    public static Tensor scatter_(Tensor input, int dim, Tensor index, Tensor src) {
        if (dim < 0)
            dim += input.shape.length;
        if (input.shape.length != index.shape.length || !java.util.Arrays.equals(index.shape, src.shape))
            throw new IllegalArgumentException("scatter_: shapes mismatch");
        int nd = input.shape.length;
        int[] inStr = computeStrides(input.shape);
        int[] idxStr = computeStrides(index.shape);
        int ne = index.numel();
        for (int linear = 0; linear < ne; linear++) {
            int rem = linear;
            int[] coord = new int[nd];
            for (int i = 0; i < nd; i++) {
                coord[i] = rem / idxStr[i];
                rem = rem % idxStr[i];
            }
            int idxAt = (int) index.data[linear];
            if (idxAt < 0)
                idxAt += input.shape[dim];
            if (idxAt < 0 || idxAt >= input.shape[dim])
                throw new IndexOutOfBoundsException("scatter_: index out of range");
            int inOff = 0;
            for (int i = 0; i < nd; i++) {
                int c = (i == dim) ? idxAt : coord[i];
                inOff += c * inStr[i];
            }
            input.data[inOff] = src.data[linear];
        }
        return input;
    }

    // activations
    public static Tensor relu(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = a.data[i] > 0 ? a.data[i] : 0f;

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++) {
                        ga.data[i] = a.data[i] > 0 ? outGrad.data[i] : 0f;
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor sigmoid(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) (1.0 / (1.0 + Math.exp(-a.data[i])));

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++) {
                        float o = out.data[i]; // sigmoid output
                        ga.data[i] = outGrad.data[i] * o * (1 - o);
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor tanh(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.tanh(a.data[i]);

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++) {
                        float o = out.data[i]; // tanh output
                        ga.data[i] = outGrad.data[i] * (1 - o * o);
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    // convert between nn.Mat (2D) and Tensor
    public static Tensor fromMat(nn.Mat m) {
        return new Tensor(m.es.clone(), m.rows, m.cols);
    }

    public static nn.Mat toMat(Tensor t) {
        if (t.shape.length != 2)
            throw new IllegalArgumentException("toMat requires 2D tensor");
        nn outer = new nn();
        nn.Mat m = outer.mat_alloc(t.shape[0], t.shape[1]);
        System.arraycopy(t.data, 0, m.es, 0, t.data.length);
        return m;
    }

    // argmax along axis 1 (rows)
    public static int[] argmax(Tensor a, int axis) {
        if (axis != 1 || a.shape.length != 2)
            throw new UnsupportedOperationException("only axis=1 for 2D implemented");
        int rows = a.shape[0], cols = a.shape[1];
        int[] out = new int[rows];
        for (int i = 0; i < rows; i++) {
            int best = 0;
            float bv = a.data[i * cols];
            for (int j = 1; j < cols; j++) {
                float v = a.data[i * cols + j];
                if (v > bv) {
                    bv = v;
                    best = j;
                }
            }
            out[i] = best;
        }
        return out;
    }

    // ----- Additional helpers implemented -----
    public static boolean is_storage(Object o) {
        return false;
    }

    public static boolean is_complex(Tensor t) {
        return false;
    }

    // More creation helpers
    public static Tensor tensor(double[] data, int... shape) {
        float[] f = new float[data.length];
        for (int i = 0; i < data.length; i++)
            f[i] = (float) data[i];
        return new Tensor(f, shape);
    }

    public static Tensor tensor(int[] data, int... shape) {
        float[] f = new float[data.length];
        for (int i = 0; i < data.length; i++)
            f[i] = data[i];
        return new Tensor(f, shape);
    }

    public static Tensor tensor(java.util.List<Float> list, int... shape) {
        float[] f = new float[list.size()];
        for (int i = 0; i < list.size(); i++)
            f[i] = list.get(i);
        return new Tensor(f, shape);
    }

    public static Tensor tensorFromIntList(java.util.List<Integer> list, int... shape) {
        float[] f = new float[list.size()];
        for (int i = 0; i < list.size(); i++)
            f[i] = list.get(i);
        return new Tensor(f, shape);
    }

    public static Tensor empty(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor linspace(float start, float end, int steps) {
        float[] d = new float[steps];
        if (steps == 1)
            d[0] = start;
        else {
            float step = (end - start) / (steps - 1);
            for (int i = 0; i < steps; i++)
                d[i] = start + i * step;
        }
        return new Tensor(d, steps);
    }

    public static Tensor logspace(float start, float end, int steps, float base) {
        float[] d = new float[steps];
        if (steps == 1)
            d[0] = (float) Math.pow(base, start);
        else {
            float step = (end - start) / (steps - 1);
            for (int i = 0; i < steps; i++)
                d[i] = (float) Math.pow(base, start + i * step);
        }
        return new Tensor(d, steps);
    }

    public static Tensor randint(int low, int high, int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = low + globalR.nextInt(high - low);
        return t;
    }

    // unary math helpers
    public static Tensor sin(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.sin(a.data[i]);
        return out;
    }

    public static Tensor cos(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.cos(a.data[i]);
        return out;
    }

    public static Tensor tan(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.tan(a.data[i]);
        return out;
    }

    public static Tensor exp(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.exp(a.data[i]);
        return out;
    }

    public static Tensor log(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.log(a.data[i]);
        return out;
    }

    public static Tensor ceil(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.ceil(a.data[i]);
        return out;
    }

    public static Tensor floor(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.floor(a.data[i]);
        return out;
    }

    public static Tensor round(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.round(a.data[i]);
        return out;
    }

    // additional trig / math
    public static Tensor asin(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.asin(a.data[i]);
        return out;
    }

    public static Tensor acos(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.acos(a.data[i]);
        return out;
    }

    public static Tensor atan(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.atan(a.data[i]);
        return out;
    }

    public static Tensor log10(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.log10(a.data[i]);
        return out;
    }

    public static Tensor log2(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) (Math.log(a.data[i]) / Math.log(2.0));
        return out;
    }

    public static Tensor trunc(Tensor a) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++) {
            float v = a.data[i];
            out.data[i] = v >= 0 ? (float) Math.floor(v) : (float) Math.ceil(v);
        }
        return out;
    }

    // comparisons: greater-equal, less-equal
    public static Tensor ge(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x >= y ? 1f : 0f);
    }

    public static Tensor le(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x <= y ? 1f : 0f);
    }

    // power
    public static Tensor pow(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> (float) Math.pow(x, y));
    }

    public static Tensor pow(Tensor a, float exp) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.pow(a.data[i], exp);
        return out;
    }

    // comparisons -> 0/1 float
    public static Tensor eq(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x == y ? 1f : 0f);
    }

    public static Tensor ne(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x != y ? 1f : 0f);
    }

    public static Tensor gt(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x > y ? 1f : 0f);
    }

    public static Tensor lt(Tensor a, Tensor b) {
        return binaryOp(a, b, (x, y) -> x < y ? 1f : 0f);
    }

    // more reductions
    public static float prod(Tensor a) {
        float p = 1f;
        for (float v : a.data)
            p *= v;
        return p;
    }

    public static Tensor max(Tensor a, int axis) {
        if (axis < 0) {
            float m = Float.NEGATIVE_INFINITY;
            for (float v : a.data)
                if (v > m)
                    m = v;
            return new Tensor(new float[] { m }, 1);
        }
        if (a.shape.length == 2 && axis == 1) {
            int r = a.shape[0], c = a.shape[1];
            Tensor out = new Tensor(r, 1);
            for (int i = 0; i < r; i++) {
                float m = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < c; j++) {
                    float v = a.data[i * c + j];
                    if (v > m)
                        m = v;
                }
                out.data[i] = m;
            }
            return out;
        }
        throw new UnsupportedOperationException("max axis not implemented");
    }

    public static Tensor min(Tensor a, int axis) {
        if (axis < 0) {
            float m = Float.POSITIVE_INFINITY;
            for (float v : a.data)
                if (v < m)
                    m = v;
            return new Tensor(new float[] { m }, 1);
        }
        if (a.shape.length == 2 && axis == 1) {
            int r = a.shape[0], c = a.shape[1];
            Tensor out = new Tensor(r, 1);
            for (int i = 0; i < r; i++) {
                float m = Float.POSITIVE_INFINITY;
                for (int j = 0; j < c; j++) {
                    float v = a.data[i * c + j];
                    if (v < m)
                        m = v;
                }
                out.data[i] = m;
            }
            return out;
        }
        throw new UnsupportedOperationException("min axis not implemented");
    }

    // variance and std (population, ddof=0)
    public static float var(Tensor a) {
        float m = mean(a);
        float s = 0f;
        for (float v : a.data) {
            float d = v - m;
            s += d * d;
        }
        return s / a.numel();
    }

    public static float std(Tensor a) {
        return (float) Math.sqrt(var(a));
    }

    // argmin along axis 1 for 2D
    public static int[] argmin(Tensor a, int axis) {
        if (axis != 1 || a.shape.length != 2)
            throw new UnsupportedOperationException("only axis=1 for 2D implemented");
        int rows = a.shape[0], cols = a.shape[1];
        int[] out = new int[rows];
        for (int i = 0; i < rows; i++) {
            int best = 0;
            float bv = a.data[i * cols];
            for (int j = 1; j < cols; j++) {
                float v = a.data[i * cols + j];
                if (v < bv) {
                    bv = v;
                    best = j;
                }
            }
            out[i] = best;
        }
        return out;
    }

    // norm: default 2-norm over all elements, or per-row when axis==1 for 2D
    public static float norm(Tensor a) {
        double s = 0.0;
        for (float v : a.data)
            s += Math.pow(Math.abs(v), 2.0);
        return (float) Math.sqrt(s);
    }

    public static Tensor norm(Tensor a, int p, int axis) {
        if (axis < 0)
            throw new UnsupportedOperationException("only axis>=0 supported");
        if (a.shape.length == 2 && axis == 1) {
            int r = a.shape[0], c = a.shape[1];
            Tensor out = new Tensor(r, 1);
            for (int i = 0; i < r; i++) {
                double s = 0.0;
                for (int j = 0; j < c; j++) {
                    s += Math.pow(Math.abs(a.data[i * c + j]), p);
                }
                out.data[i] = (float) Math.pow(s, 1.0 / p);
            }
            return out;
        }
        throw new UnsupportedOperationException("norm axis not implemented for this shape");
    }

    // aliases / linear algebra
    public static Tensor mm(Tensor a, Tensor b) {
        return matmul(a, b);
    }

    public static Tensor bmm(Tensor a, Tensor b) {
        if (a.shape.length != 3 || b.shape.length != 3)
            throw new IllegalArgumentException("bmm requires 3D tensors");
        int B = a.shape[0], M = a.shape[1], K = a.shape[2], N = b.shape[2];
        if (b.shape[0] != B || b.shape[1] != K)
            throw new IllegalArgumentException("bmm shape mismatch");
        Tensor out = new Tensor(B, M, N);
        for (int bIdx = 0; bIdx < B; bIdx++) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float s = 0f;
                    for (int kIdx = 0; kIdx < K; kIdx++) {
                        float va = a.data[bIdx * M * K + i * K + kIdx];
                        float vb = b.data[bIdx * K * N + kIdx * N + j];
                        s += va * vb;
                    }
                    out.data[bIdx * M * N + i * N + j] = s;
                }
        }

        if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        // dA = outGrad bmm b.transpose(1, 2)
                        Tensor bt = transposeLastTwoDims(b);
                        Tensor ga = bmm(outGrad, bt);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        // dB = a.transpose(1, 2) bmm outGrad
                        Tensor at = transposeLastTwoDims(a);
                        Tensor gb = bmm(at, outGrad);
                        b.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    private static Tensor transposeLastTwoDims(Tensor t) {
        int nd = t.shape.length;
        if (nd < 2)
            return t; // degenerate
        int[] dims = new int[nd];
        for (int i = 0; i < nd; i++)
            dims[i] = i;
        dims[nd - 2] = nd - 1;
        dims[nd - 1] = nd - 2;
        return permute(t, dims);
    }

    // Vector Species for SIMD optimizations
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static float dot(Tensor a, Tensor b) {
        if (a.numel() != b.numel())
            throw new IllegalArgumentException("dot size mismatch");

        int length = a.data.length;
        int i = 0;
        float sum = 0f;
        int upperBound = SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.zero(SPECIES);

        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a.data, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b.data, i);
            sumVector = sumVector.add(va.mul(vb));
        }

        // Sum values across the vector lanes
        sum += sumVector.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);

        // Tail loop for remaining elements
        for (; i < length; i++) {
            sum += a.data[i] * b.data[i];
        }
        return sum;
    }

    // matrix inverse (Gauss-Jordan) for 2D square tensors
    public static Tensor inverse(Tensor a) {
        if (a.shape.length != 2)
            throw new IllegalArgumentException("inverse requires 2D square tensor");
        int n = a.shape[0];
        if (a.shape[1] != n)
            throw new IllegalArgumentException("inverse requires square matrix");
        // copy to double for numerical stability
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i][j] = a.data[i * n + j];
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++)
            I[i][i] = 1.0;

        for (int col = 0; col < n; col++) {
            // partial pivot
            int piv = col;
            double maxv = Math.abs(M[col][col]);
            for (int r = col + 1; r < n; r++) {
                double av = Math.abs(M[r][col]);
                if (av > maxv) {
                    maxv = av;
                    piv = r;
                }
            }
            if (Math.abs(M[piv][col]) < 1e-12)
                throw new IllegalArgumentException("matrix is singular");
            if (piv != col) {
                double[] tmp = M[col];
                M[col] = M[piv];
                M[piv] = tmp;
                double[] t2 = I[col];
                I[col] = I[piv];
                I[piv] = t2;
            }
            double diag = M[col][col];
            // normalize row
            for (int j = 0; j < n; j++) {
                M[col][j] /= diag;
                I[col][j] /= diag;
            }
            // eliminate other rows
            for (int r = 0; r < n; r++) {
                if (r == col)
                    continue;
                double factor = M[r][col];
                if (factor == 0.0)
                    continue;
                for (int j = 0; j < n; j++) {
                    M[r][j] -= factor * M[col][j];
                    I[r][j] -= factor * I[col][j];
                }
            }
        }
        float[] out = new float[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                out[i * n + j] = (float) I[i][j];
        return new Tensor(out, n, n);
    }

    // determinant via Gaussian elimination with partial pivoting
    public static float det(Tensor a) {
        if (a.shape.length != 2)
            throw new IllegalArgumentException("det requires 2D square tensor");
        int n = a.shape[0];
        if (a.shape[1] != n)
            throw new IllegalArgumentException("det requires square matrix");
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i][j] = a.data[i * n + j];
        int swaps = 0;
        double det = 1.0;
        for (int col = 0; col < n; col++) {
            int piv = col;
            double maxv = Math.abs(M[col][col]);
            for (int r = col + 1; r < n; r++) {
                double av = Math.abs(M[r][col]);
                if (av > maxv) {
                    maxv = av;
                    piv = r;
                }
            }
            if (Math.abs(M[piv][col]) < 1e-12)
                return 0f;
            if (piv != col) {
                double[] tmp = M[col];
                M[col] = M[piv];
                M[piv] = tmp;
                swaps++;
            }
            double diag = M[col][col];
            det *= diag;
            // eliminate below
            for (int r = col + 1; r < n; r++) {
                double factor = M[r][col] / diag;
                for (int j = col; j < n; j++)
                    M[r][j] -= factor * M[col][j];
            }
        }
        if ((swaps & 1) == 1)
            det = -det;
        return (float) det;
    }

    // save/load simple text format
    public static void save(Tensor t, String path) throws java.io.IOException {
        java.nio.file.Files.createDirectories(java.nio.file.Paths.get(path).getParent());
        try (java.io.BufferedWriter bw = java.nio.file.Files.newBufferedWriter(java.nio.file.Paths.get(path))) {
            bw.write(Integer.toString(t.shape.length));
            bw.newLine();
            for (int i = 0; i < t.shape.length; i++) {
                if (i > 0)
                    bw.write(",");
                bw.write(Integer.toString(t.shape[i]));
            }
            bw.newLine();
            for (int i = 0; i < t.data.length; i++) {
                bw.write(Float.toString(t.data[i]));
                if (i + 1 < t.data.length)
                    bw.write(",");
            }
            bw.newLine();
        }
    }

    public static Tensor load(String path) throws java.io.IOException {
        java.util.List<String> lines = java.nio.file.Files.readAllLines(java.nio.file.Paths.get(path));
        if (lines.size() < 3)
            throw new java.io.IOException("invalid tensor file");
        int nd = Integer.parseInt(lines.get(0).trim());
        String[] dims = lines.get(1).trim().split(",");
        int[] shape = new int[nd];
        for (int i = 0; i < nd; i++)
            shape[i] = Integer.parseInt(dims[i]);
        String[] vals = lines.get(2).trim().split(",");
        float[] data = new float[vals.length];
        for (int i = 0; i < vals.length; i++)
            data[i] = Float.parseFloat(vals[i]);
        return new Tensor(data, shape);
    }

    // gradient mode stubs
    private static boolean gradEnabled = true;

    public static java.io.Closeable no_grad() {
        gradEnabled = false;
        return () -> {
            gradEnabled = true;
        };
    }

    public static void enable_grad() {
        gradEnabled = true;
    }

    public static void set_grad_enabled(boolean enabled) {
        gradEnabled = enabled;
    }

    public static boolean is_grad_enabled() {
        return gradEnabled;
    }

    // cat(tensors, dim): Concatenates the given sequence of tensors in the given
    // dimension.
    public static Tensor cat(java.util.List<Tensor> tensors, int dim) {
        if (tensors.isEmpty())
            throw new IllegalArgumentException("cat requires at least one tensor");
        if (dim < 0)
            dim += tensors.get(0).shape.length;

        // Validate shapes
        int[] baseShape = tensors.get(0).shape;
        for (int i = 0; i < baseShape.length; i++) {
            if (i == dim)
                continue;
            for (int j = 1; j < tensors.size(); j++) {
                if (tensors.get(j).shape[i] != baseShape[i])
                    throw new IllegalArgumentException("cat: Tensors must have same shape except in the cat dimension");
            }
        }

        int totalDimSize = 0;
        for (Tensor t : tensors) {
            totalDimSize += t.shape[dim];
        }

        int[] newShape = baseShape.clone();
        newShape[dim] = totalDimSize;
        Tensor res = new Tensor(newShape);

        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= baseShape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < baseShape.length; i++)
            innerSize *= baseShape[i];

        int currentOffset = 0;
        for (Tensor t : tensors) {
            int tDimSize = t.shape[dim];
            for (int i = 0; i < outerSize; i++) {
                System.arraycopy(
                        t.data, i * tDimSize * innerSize,
                        res.data, (i * totalDimSize + currentOffset) * innerSize,
                        tDimSize * innerSize);
            }
            currentOffset += tDimSize;
        }

        if (is_grad_enabled()) {
            boolean anyGrad = false;
            for (Tensor t : tensors)
                if (t.requires_grad) {
                    anyGrad = true;
                    break;
                }
            if (anyGrad) {
                final int finalDim = dim;
                res.requires_grad = true;
                res.grad_fn = new Tensor.GradFn(tensors.toArray(new Tensor[0])) {
                    public void apply(Tensor outGrad) {
                        int p = 0;
                        for (Tensor t : tensors) {
                            int tDimSize = t.shape[finalDim];
                            if (t.requires_grad) {
                                Tensor grad = narrow(outGrad, finalDim, p, tDimSize);
                                t.backwardStep(grad);
                            }
                            p += tDimSize;
                        }
                    }
                };
            }
        }
        return res;
    }

    // narrow(input, dim, start, length): Returns a new tensor that is a narrowed
    // version of input tensor.
    public static Tensor narrow(Tensor input, int dim, int start, int length) {
        if (dim < 0)
            dim += input.shape.length;
        int[] newShape = input.shape.clone();
        newShape[dim] = length;
        Tensor out = new Tensor(newShape);

        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= input.shape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < input.shape.length; i++)
            innerSize *= input.shape[i];

        int oldDimSize = input.shape[dim];

        final int finalDim = dim;
        final int finalStart = start;
        final int finalLength = length;
        final int finalInnerSize = innerSize;
        final int finalOuterSize = outerSize;
        final int finalOldDimSize = input.shape[dim];

        for (int i = 0; i < outerSize; i++) {
            // Copy a block for each outer index
            System.arraycopy(
                    input.data, (i * finalOldDimSize + finalStart) * finalInnerSize,
                    out.data, (i * finalLength) * finalInnerSize,
                    finalLength * finalInnerSize);
        }

        if (is_grad_enabled() && input.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(input) {
                public void apply(Tensor outGrad) {
                    Tensor grad = new Tensor(input.shape);
                    for (int i = 0; i < finalOuterSize; i++) {
                        System.arraycopy(
                                outGrad.data, (i * finalLength) * finalInnerSize,
                                grad.data, (i * finalOldDimSize + finalStart) * finalInnerSize,
                                finalLength * finalInnerSize);
                    }
                    input.backwardStep(grad);
                }
            };
        }
        return out;
    }

}
