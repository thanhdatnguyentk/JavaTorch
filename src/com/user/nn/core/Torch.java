package com.user.nn.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

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
        Tensor out = binaryOp(a, b, (x, y) -> x - y, BinaryOpType.SUB);
        if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor ga = reduceSumToShape(outGrad, a.shape);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        Tensor gb = binaryOp(outGrad, full(outGrad.shape, -1.0f), (x, y) -> x * y);
                        gb = reduceSumToShape(gb, b.shape);
                        b.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    public static Tensor mul(Tensor a, Tensor b) {
        Tensor out = binaryOp(a, b, (x, y) -> x * y, BinaryOpType.MUL);
        // autograd for mul: dOut/dA = B * outGrad ; dOut/dB = A * outGrad
        if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor ga = binaryOp(outGrad, b, (x, y) -> x * y, BinaryOpType.MUL); // outGrad * b
                        ga = reduceSumToShape(ga, a.shape);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        Tensor gb = binaryOp(outGrad, a, (x, y) -> x * y, BinaryOpType.MUL);
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
        Tensor out = binaryOp(a, b, (x, y) -> x + y, BinaryOpType.ADD);
        if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
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
        return binaryOp(a, b, (x, y) -> x / y, BinaryOpType.DIV);
    }

    // scalar variants
    public static Tensor add(Tensor a, float scalar) {
        if (a.isGPU()) {
            Tensor out = new Tensor(a.shape);
            out.toGPU();
            CUDAOps.add(a, scalar, out);
            if (is_grad_enabled() && a.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(a) {
                    public void apply(Tensor outGrad) {
                        a.backwardStep(outGrad.clone());
                    }
                };
            }
            return out;
        }

        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = a.data[i] + scalar;
        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    a.backwardStep(outGrad.clone());
                }
            };
        }
        return out;
    }

    public static Tensor mul(Tensor a, float scalar) {
        if (a.isGPU()) {
            Tensor out = new Tensor(a.shape);
            out.toGPU();
            CUDAOps.mul(a, scalar, out);
            if (is_grad_enabled() && a.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(a) {
                    public void apply(Tensor outGrad) {
                        a.backwardStep(mul(outGrad, scalar));
                    }
                };
            }
            return out;
        }
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = a.data[i] * scalar;
        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int j = 0; j < ga.data.length; j++)
                        ga.data[j] = outGrad.data[j] * scalar;
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor div(Tensor a, float scalar) {
        return mul(a, 1.0f / scalar);
    }

    public static Tensor sub(Tensor a, float scalar) {
        return add(a, -scalar);
    }

    // permute dims
    public static Tensor permute(Tensor a, int... dims) {
        int ndOrigin = a.shape.length;
        if (dims.length != ndOrigin) throw new IllegalArgumentException("permute dims length mismatch");
        
        // Check if it's already in order
        boolean same = true;
        for (int i = 0; i < dims.length; i++) if (dims[i] != i) same = false;
        if (same) return a.clone();

        // One-time generic transpose using the coordinate logic
        a.toCPU();
        int[] outShape = new int[ndOrigin];
        for (int i = 0; i < ndOrigin; i++) outShape[i] = a.shape[dims[i]];
        
        Tensor out = new Tensor(outShape);
        int[] aStrides = computeStrides(a.shape);
        int[] outStrides = computeStrides(outShape);
        
        for (int i = 0; i < a.numel(); i++) {
            int[] coords = getCoords(i, aStrides);
            int[] newCoords = new int[ndOrigin];
            for (int d = 0; d < ndOrigin; d++) newCoords[d] = coords[dims[d]];
            
            int outIdx = getIndex(newCoords, outStrides);
            out.data[outIdx] = a.data[i];
        }
        
        if (a.isGPU()) out.toGPU();
        
        if (is_grad_enabled() && a.requires_grad) {
            // Gradient of permute(dims) is permute(inverse_dims)
            int[] invDims = new int[ndOrigin];
            for (int i = 0; i < ndOrigin; i++) invDims[dims[i]] = i;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor gradOutput) {
                    a.backwardStep(permute(gradOutput, invDims));
                }
            };
        }
        return out;
    }


    public static Tensor sub(float scalar, Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = scalar - a.data[i];
        if (a.isGPU()) out.toGPU();
        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = mul(outGrad, -1.0f);
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    // public add that picks autograd-aware impl
    public static Tensor add(Tensor a, Tensor b) {
        return addWithGrad(a, b);
    }

    // scalar add/mul autograd not implemented (user can use elementwise ops with
    // tensors)

    public enum BinaryOpType { ADD, SUB, MUL, DIV, OTHER }

    private static Tensor binaryOp(Tensor a, Tensor b, FloatBinaryOp op) {
        return binaryOp(a, b, op, BinaryOpType.OTHER);
    }

    private static Tensor binaryOp(Tensor a, Tensor b, FloatBinaryOp op, BinaryOpType type) {
        // Fast path for exact shape match on GPU without broadcasting
        if (type != BinaryOpType.OTHER && type != BinaryOpType.DIV 
                && a.isGPU() && b.isGPU() && java.util.Arrays.equals(a.shape, b.shape)) {
            Tensor out = new Tensor(a.shape).toGPU();
            if (type == BinaryOpType.ADD) {
                CUDAOps.add(a, b, out);
            } else if (type == BinaryOpType.SUB) {
                CUDAOps.sub(a, b, out);
            } else if (type == BinaryOpType.MUL) {
                CUDAOps.mul(a, b, out);
            }
            return out;
        }

        a.toCPU();
        b.toCPU();
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
        if (a.isGPU() || b.isGPU()) out.toGPU();
        return out;
    }

    // Reduce outGrad (of shape grad.shape) to targetShape by summing broadcasted
    // dims
    private static Tensor reduceSumToShape(Tensor grad, int[] targetShape) {
        if (java.util.Arrays.equals(grad.shape, targetShape))
            return grad;
        grad.toCPU();
        Tensor out = new Tensor(targetShape);
        int gn = grad.shape.length, tn = targetShape.length;
        int nout = Math.max(gn, tn);
        int[] gs2 = new int[nout], ts2 = new int[nout];
        java.util.Arrays.fill(gs2, 1);
        java.util.Arrays.fill(ts2, 1);
        System.arraycopy(grad.shape, 0, gs2, nout - gn, gn);
        System.arraycopy(targetShape, 0, ts2, nout - tn, tn);
        int[] gStr = computeStrides(gs2);
        int[] tStr = computeStrides(ts2);
        int total = grad.numel();
        for (int i = 0; i < total; i++) {
            int rem = i, outLinear = 0;
            for (int d = 0; d < nout; d++) {
                int c = rem / gStr[d];
                rem %= gStr[d];
                if (ts2[d] != 1) outLinear += c * tStr[d];
            }
            out.data[outLinear] += grad.data[i];
        }
        if (grad.isGPU()) out.toGPU();
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
        if (a.isGPU()) {
            Tensor out = new Tensor(1).toGPU();
            CUDAOps.reduceSum(a, out);
            return out.data[0];
        }
        float s = 0f;
        for (float v : a.data)
            s += v;
        return s;
    }

    public static float mean(Tensor a) {
        return sum(a) / a.numel();
    }

    public static Tensor sum_tensor(Tensor a) {
        Tensor out = new Tensor(1);
        if (a.isGPU()) {
            out.toGPU();
            CUDAOps.reduceSum(a, out);
        } else {
            float s = 0f;
            for (float v : a.data) s += v;
            out.data[0] = s;
        }
        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor gradOutput) {
                    // Gradient of sum is ones of shape a * gradOutput.item()
                    Tensor ga = ones(a.shape);
                    if (a.isGPU()) ga.toGPU();
                    ga = mul(ga, gradOutput.item());
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor mean_tensor(Tensor a) {
        Tensor s = sum_tensor(a);
        return mul(s, 1.0f / a.numel());
    }

    // --- Batch 3: Pooling and Padding ---

    public static Tensor max_pool1d(Tensor x, int kernel, int stride, int pad) {
        x.toCPU();
        int nd = x.shape.length;
        if (nd < 2)
            throw new IllegalArgumentException("Expected x to have at least 2 dims [C, L] or [N, C, L]");
        final int batch = (nd == 3) ? x.shape[0] : 1;
        final int inC = x.shape[nd - 2];
        final int inL = x.shape[nd - 1];
        final int outL = (inL + 2 * pad - kernel) / stride + 1;
        int outShape[] = (nd == 3) ? new int[] { batch, inC, outL } : new int[] { inC, outL };
        Tensor out = new Tensor(outShape);
        if (x.isGPU()) out.toGPU();
        final int[] maxIndices = new int[out.numel()];

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < inC; c++) {
                for (int ol = 0; ol < outL; ol++) {
                    float maxv = Float.NEGATIVE_INFINITY;
                    int maxIdx = -1;
                    for (int k = 0; k < kernel; k++) {
                        int il = ol * stride - pad + k;
                        if (il >= 0 && il < inL) {
                            int idx = (nd == 3) ? (b * inC * inL + c * inL + il) : (c * inL + il);
                            float v = x.data[idx];
                            if (v > maxv) {
                                maxv = v;
                                maxIdx = idx;
                            }
                        }
                    }
                    int outIdx = (nd == 3) ? (b * inC * outL + c * outL + ol) : (c * outL + ol);
                    out.data[outIdx] = maxv;
                    maxIndices[outIdx] = maxIdx;
                }
            }
        }
        if (is_grad_enabled() && x.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x) {
                public void apply(Tensor outGrad) {
                    Tensor gx = new Tensor(x.shape);
                    for (int i = 0; i < outGrad.data.length; i++) {
                        if (maxIndices[i] >= 0)
                            gx.data[maxIndices[i]] += outGrad.data[i];
                    }
                    x.backwardStep(gx);
                }
            };
        }
        return out;
    }

    public static Tensor avg_pool1d(Tensor x, int kernel, int stride, int pad) {
        x.toCPU();
        int nd = x.shape.length;
        if (nd < 2)
            throw new IllegalArgumentException("Expected x to have at least 2 dims");
        final int batch = (nd == 3) ? x.shape[0] : 1;
        final int inC = x.shape[nd - 2];
        final int inL = x.shape[nd - 1];
        final int outL = (inL + 2 * pad - kernel) / stride + 1;
        int outShape[] = (nd == 3) ? new int[] { batch, inC, outL } : new int[] { inC, outL };
        Tensor out = new Tensor(outShape);
        if (x.isGPU()) out.toGPU();
        final float[] counts = new float[out.numel()];

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < inC; c++) {
                for (int ol = 0; ol < outL; ol++) {
                    float sum = 0f;
                    int cnt = 0;
                    for (int k = 0; k < kernel; k++) {
                        int il = ol * stride - pad + k;
                        if (il >= 0 && il < inL) {
                            sum += x.data[(nd == 3) ? (b * inC * inL + c * inL + il) : (c * inL + il)];
                            cnt++;
                        }
                    }
                    int outIdx = (nd == 3) ? (b * inC * outL + c * outL + ol) : (c * outL + ol);
                    out.data[outIdx] = cnt > 0 ? sum / cnt : 0f;
                    counts[outIdx] = (float) cnt;
                }
            }
        }
        if (is_grad_enabled() && x.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x) {
                public void apply(Tensor outGrad) {
                    Tensor gx = new Tensor(x.shape);
                    for (int b = 0; b < batch; b++) {
                        for (int c = 0; c < inC; c++) {
                            for (int ol = 0; ol < outL; ol++) {
                                int outIdx = (nd == 3) ? (b * inC * outL + c * outL + ol) : (c * outL + ol);
                                float cnt = counts[outIdx];
                                if (cnt > 0) {
                                    float grad = outGrad.data[outIdx] / cnt;
                                    for (int k = 0; k < kernel; k++) {
                                        int il = ol * stride - pad + k;
                                        if (il >= 0 && il < inL) {
                                            int idx = (nd == 3) ? (b * inC * inL + c * inL + il) : (c * inL + il);
                                            gx.data[idx] += grad;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    x.backwardStep(gx);
                }
            };
        }
        return out;
    }

    public static Tensor adaptive_avg_pool2d(Tensor x, int[] outputSize) {
        x.toCPU();
        final int nd = x.shape.length;
        if (nd < 3)
            throw new IllegalArgumentException("Expected x to have at least 3 dims [C, H, W] or [N, C, H, W]");
        final int batch = (nd == 4) ? x.shape[0] : 1;
        final int inC = x.shape[nd - 3];
        final int inH = x.shape[nd - 2];
        final int inW = x.shape[nd - 1];
        final int outH = outputSize[0];
        final int outW = outputSize[1];
        int outShape[] = (nd == 4) ? new int[] { batch, inC, outH, outW } : new int[] { inC, outH, outW };
        Tensor out = new Tensor(outShape);
        if (x.isGPU()) out.toGPU();

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < inC; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    int hs = (int) Math.floor((double) oh * inH / outH);
                    int he = (int) Math.ceil((double) (oh + 1) * inH / outH);
                    for (int ow = 0; ow < outW; ow++) {
                        int ws = (int) Math.floor((double) ow * inW / outW);
                        int we = (int) Math.ceil((double) (ow + 1) * inW / outW);
                        float sum = 0;
                        int cnt = 0;
                        for (int h = hs; h < he; h++) {
                            for (int w = ws; w < we; w++) {
                                sum += x.data[(nd == 4) ? (b * inC * inH * inW + c * inH * inW + h * inW + w)
                                        : (c * inH * inW + h * inW + w)];
                                cnt++;
                            }
                        }
                        int outIdx = (nd == 4) ? (b * inC * outH * outW + c * outH * outW + oh * outW + ow)
                                : (c * outH * outW + oh * outW + ow);
                        out.data[outIdx] = cnt > 0 ? sum / cnt : 0f;
                    }
                }
            }
        }
        if (is_grad_enabled() && x.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x) {
                public void apply(Tensor outGrad) {
                    outGrad.toCPU();
                    Tensor gx = new Tensor(x.shape);
                    for (int b = 0; b < batch; b++) {
                        for (int c = 0; c < inC; c++) {
                            for (int oh = 0; oh < outH; oh++) {
                                int hs = (int) Math.floor((double) oh * inH / outH);
                                int he = (int) Math.ceil((double) (oh + 1) * inH / outH);
                                int hSize = he - hs;
                                for (int ow = 0; ow < outW; ow++) {
                                    int ws = (int) Math.floor((double) ow * inW / outW);
                                    int we = (int) Math.ceil((double) (ow + 1) * inW / outW);
                                    int wSize = we - ws;
                                    int outIdx = (nd == 4) ? (b * inC * outH * outW + c * outH * outW + oh * outW + ow)
                                            : (c * outH * outW + oh * outW + ow);
                                    float g = outGrad.data[outIdx] / (hSize * wSize);
                                    for (int h = hs; h < he; h++) {
                                        for (int w = ws; w < we; w++) {
                                            int idx = (nd == 4)
                                                    ? (b * inC * inH * inW + c * inH * inW + h * inW + w)
                                                    : (c * inH * inW + h * inW + w);
                                            gx.data[idx] += g;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    x.backwardStep(gx);
                }
            };
        }
        return out;
    }

    public static Tensor pad(Tensor x, final int[] pad, String mode, float value) {
        if (!mode.equals("constant")) {
            throw new UnsupportedOperationException("Only constant padding supported");
        }
        final int nd = x.shape.length;
        int[] newShape = x.shape.clone();
        for (int i = 0; i < pad.length / 2; i++) {
            int dim = nd - 1 - i;
            if (dim < 0)
                break;
            newShape[dim] += pad[i * 2] + pad[i * 2 + 1];
        }
        Tensor out = new Tensor(newShape);
        if (value != 0f) {
            for (int i = 0; i < out.data.length; i++)
                out.data[i] = value;
        }

        copyToPad(x, out, pad, 0, new int[nd], new int[nd]);

        if (is_grad_enabled() && x.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x) {
                public void apply(Tensor outGrad) {
                    Tensor gx = new Tensor(x.shape);
                    copyFromPad(gx, outGrad, pad, 0, new int[nd], new int[nd]);
                    x.backwardStep(gx);
                }
            };
        }
        return out;
    }

    private static void copyToPad(Tensor src, Tensor dst, int[] pad, int dim, int[] srcIdx, int[] dstIdx) {
        int nd = src.shape.length;
        if (dim == nd) {
            dst.data[dst.offset(dstIdx)] = src.data[src.offset(srcIdx)];
            return;
        }
        int pIdx = (nd - 1 - dim) * 2;
        int pStart = (pad.length > pIdx) ? pad[pIdx] : 0;
        for (int i = 0; i < src.shape[dim]; i++) {
            srcIdx[dim] = i;
            dstIdx[dim] = i + pStart;
            copyToPad(src, dst, pad, dim + 1, srcIdx, dstIdx);
        }
    }

    private static void copyFromPad(Tensor srcGrad, Tensor dstGrad, int[] pad, int dim, int[] srcIdx, int[] dstIdx) {
        int nd = srcGrad.shape.length;
        if (dim == nd) {
            srcGrad.data[srcGrad.offset(srcIdx)] += dstGrad.data[dstGrad.offset(dstIdx)];
            return;
        }
        int pIdx = (nd - 1 - dim) * 2;
        int pStart = (pad.length > pIdx) ? pad[pIdx] : 0;
        for (int i = 0; i < srcGrad.shape[dim]; i++) {
            srcIdx[dim] = i;
            dstIdx[dim] = i + pStart;
            copyFromPad(srcGrad, dstGrad, pad, dim + 1, srcIdx, dstIdx);
        }
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
        int nd = a.shape.length;
        if (dim0 < 0) dim0 += nd;
        if (dim1 < 0) dim1 += nd;
        if (dim0 == dim1) return a.clone();
        
        int[] outShape = a.shape.clone();
        outShape[dim0] = a.shape[dim1];
        outShape[dim1] = a.shape[dim0];

        Tensor out;
        if (a.isGPU() && nd == 2 && dim0 == 0 && dim1 == 1) {
            out = new Tensor(outShape).toGPU();
            CUDAOps.transpose(a, out);
        } else {
            a.toCPU();
            out = new Tensor(outShape);
            int[] aStrides = computeStrides(a.shape);
            int[] outStrides = computeStrides(outShape);

            for (int i = 0; i < a.numel(); i++) {
                int[] coords = getCoords(i, aStrides);
                int tmp = coords[dim0];
                coords[dim0] = coords[dim1];
                coords[dim1] = tmp;
                out.data[getIndex(coords, outStrides)] = a.data[i];
            }
            if (a.isGPU()) out.toGPU();
        }

        if (is_grad_enabled() && a.requires_grad) {
            final int fDim0 = dim0, fDim1 = dim1;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor gradOutput) {
                    a.backwardStep(transpose(gradOutput, fDim0, fDim1));
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
        if (a.shape.length == 3 && b.shape.length == 3) {
            return bmm(a, b);
        }
        if (a.shape.length != 2 || b.shape.length != 2) {
             // Basic support for [..., M, K] @ [K, N]
             if (a.shape.length > 2 && b.shape.length == 2) {
                 int[] batch = java.util.Arrays.copyOf(a.shape, a.shape.length - 2);
                 int m = a.shape[a.shape.length - 2];
                 int k = a.shape[a.shape.length - 1];
                 int n = b.shape[1];
                 if (k != b.shape[0]) throw new IllegalArgumentException("matmul mismatch");
                 
                 int bProd = 1; for(int s : batch) bProd *= s;
                 Tensor aFlat = a.reshape(bProd, m, k);
                 // We don't have expand() yet, but we can do a loop of matmuls or a custom tile
                 // Let's use the CPU fallback if not on GPU, or a GPU loop
                 Tensor outFlat = new Tensor(bProd, m, n);
                 if (a.isGPU() || b.isGPU()) outFlat.toGPU();
                 
                 for (int i = 0; i < bProd; i++) {
                     Tensor ai = narrow(aFlat, 0, i, 1).reshape(m, k);
                     Tensor ci = matmul(ai, b);
                     if (outFlat.isGPU()) {
                         // CUDA copy ci -> outFlat[i]
                         cudaMemcpy(outFlat.getDevicePointer().withByteOffset((long)i*m*n*4), ci.getDevicePointer(), (long)m*n*4, cudaMemcpyDeviceToDevice);
                     } else {
                         System.arraycopy(ci.data, 0, outFlat.data, i * m * n, m * n);
                     }
                 }
                 int[] outShape = new int[a.shape.length];
                 System.arraycopy(a.shape, 0, outShape, 0, a.shape.length-2);
                 outShape[a.shape.length-2] = m;
                 outShape[a.shape.length-1] = n;
                 return outFlat.reshape(outShape);
             }
             throw new IllegalArgumentException("matmul supports 2D or 3D/batched cases");
        }
        
        int m = a.shape[0], k = a.shape[1], n = b.shape[1];
        if (k != b.shape[0]) throw new IllegalArgumentException("matmul shape mismatch");

        // GPU Path
        if (a.device == Tensor.Device.GPU || b.device == Tensor.Device.GPU) {
            if (!a.isGPU()) a.toGPU();
            if (!b.isGPU()) b.toGPU();
            Tensor out = new Tensor(m, n);
            out.toGPU();
            CUDAOps.matmul(a, b, out);
            
            if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(a, b) {
                    public void apply(Tensor outGrad) {
                        if (a.requires_grad) {
                            Tensor bt = transpose(b);
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

        // CPU Path — OpenBLAS (large) or SIMD (small)
        Tensor out = new Tensor(m, n);
        if (BlasOps.isAvailable() && (long) m * n * k > 4096) {
            a.toCPU(); b.toCPU();
            BlasOps.sgemm(a.data, b.data, out.data, m, n, k);
        } else {
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
        }

        if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor ga = matmul(outGrad, transpose(b));
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

    public static Tensor bmm(Tensor a, Tensor b) {
        if (a.shape.length != 3 || b.shape.length != 3) {
            throw new IllegalArgumentException("bmm supports 3D tensors [B, M, K] and [B, K, N]");
        }
        int bSize = a.shape[0];
        int m = a.shape[1], k = a.shape[2];
        if (b.shape[0] != bSize || b.shape[1] != k) {
            throw new IllegalArgumentException("bmm shape mismatch: a=" + java.util.Arrays.toString(a.shape) + " b=" + java.util.Arrays.toString(b.shape));
        }
        int n = b.shape[2];

        // GPU Path
        if (a.device == Tensor.Device.GPU || b.device == Tensor.Device.GPU) {
            if (!a.isGPU()) a.toGPU();
            if (!b.isGPU()) b.toGPU();
            Tensor out = new Tensor(bSize, m, n);
            out.toGPU();
            CUDAOps.bmm(a, b, out);
            
            if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(a, b) {
                    public void apply(Tensor outGrad) {
                        if (a.requires_grad) {
                            // dA = dC @ B.T
                            Tensor bt = transpose(b, 1, 2);
                            Tensor ga = bmm(outGrad, bt);
                            a.backwardStep(ga);
                        }
                        if (b.requires_grad) {
                            // dB = A.T @ dC
                            Tensor at = transpose(a, 1, 2);
                            Tensor gb = bmm(at, outGrad);
                            b.backwardStep(gb);
                        }
                    }
                };
            }
            return out;
        }

        // CPU Fallback
        Tensor out = new Tensor(bSize, m, n);
        for (int i = 0; i < bSize; i++) {
            Tensor ai = narrow(a, 0, i, 1).reshape(m, k);
            Tensor bi = narrow(b, 0, i, 1).reshape(k, n);
            Tensor ci = matmul(ai, bi);
            System.arraycopy(ci.data, 0, out.data, i * m * n, m * n);
        }
        
        if (is_grad_enabled() && (a.requires_grad || b.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a, b) {
                public void apply(Tensor outGrad) {
                    if (a.requires_grad) {
                        Tensor bt = transpose(b, 1, 2);
                        Tensor ga = bmm(outGrad, bt);
                        a.backwardStep(ga);
                    }
                    if (b.requires_grad) {
                        Tensor at = transpose(a, 1, 2);
                        Tensor gb = bmm(at, outGrad);
                        b.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    private static int[] getCoords(int idx, int[] strides) {
        int n = strides.length;
        int[] coords = new int[n];
        int rem = idx;
        for (int i = 0; i < n; i++) {
            coords[i] = rem / strides[i];
            rem = rem % strides[i];
        }
        return coords;
    }

    private static int getIndex(int[] coords, int[] strides) {
        int idx = 0;
        for (int i = 0; i < coords.length; i++) {
            idx += coords[i] * strides[i];
        }
        return idx;
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

    // expand: repeat tensor along singleton dimensions
    public static Tensor expand(Tensor a, int... newShape) {
        // Use broadcasting hack: add to zeros of target shape
        Tensor z = zeros(newShape).to(a.device);
        return add(a, z);
    }

    // where: choose elements from x or y based on condition (cond != 0)
    public static Tensor where(Tensor cond, Tensor x, Tensor y) {
        if (cond.numel() != x.numel() || x.numel() != y.numel())
            throw new IllegalArgumentException("where: shapes must match elementwise");
        cond.toCPU();
        x.toCPU();
        y.toCPU();
        Tensor out = new Tensor(x.shape);
        for (int i = 0; i < x.data.length; i++)
            out.data[i] = (cond.data[i] != 0f) ? x.data[i] : y.data[i];
        if (cond.isGPU() || x.isGPU() || y.isGPU()) out.toGPU();
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
        input.toCPU();
        index.toCPU();
        int nd = input.shape.length;
        int[] inStr = computeStrides(input.shape);
        int[] idxStr = computeStrides(index.shape);
        Tensor out = new Tensor(index.shape);
        if (input.isGPU() || index.isGPU()) out.toGPU();
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
        input.toCPU();
        index.toCPU();
        src.toCPU();
        int nd = input.shape.length;
        int[] inStr = computeStrides(input.shape);
        int[] idxStr = computeStrides(index.shape);
        Tensor out = input.clone();
        if (input.isGPU() || index.isGPU() || src.isGPU()) out.toGPU();
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
        if (a.isGPU()) {
            out.toGPU();
            CUDAOps.reluForward(a, out);
        } else {
            for (int i = 0; i < a.data.length; i++)
                out.data[i] = a.data[i] > 0 ? a.data[i] : 0f;
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    if (a.isGPU()) {
                        ga.toGPU();
                        CUDAOps.reluBackward(a, outGrad, ga);
                    } else {
                        for (int i = 0; i < ga.data.length; i++) {
                            ga.data[i] = a.data[i] > 0 ? outGrad.data[i] : 0f;
                        }
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor sigmoid(Tensor a) {
        Tensor out = new Tensor(a.shape);
        if (a.isGPU()) {
            out.toGPU();
            CUDAOps.sigmoidForward(a, out);
        } else {
            for (int i = 0; i < a.data.length; i++)
                out.data[i] = (float) (1.0 / (1.0 + Math.exp(-a.data[i])));
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    if (a.isGPU()) {
                        ga.toGPU();
                        CUDAOps.sigmoidBackward(out, outGrad, ga);
                    } else {
                        for (int i = 0; i < ga.data.length; i++) {
                            float o = out.data[i]; // sigmoid output
                            ga.data[i] = outGrad.data[i] * o * (1 - o);
                        }
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor tanh(Tensor a) {
        Tensor out = new Tensor(a.shape);
        if (a.isGPU()) {
            out.toGPU();
            CUDAOps.tanhForward(a, out);
        } else {
            for (int i = 0; i < a.data.length; i++)
                out.data[i] = (float) Math.tanh(a.data[i]);
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    if (a.isGPU()) {
                        ga.toGPU();
                        CUDAOps.tanhBackward(out, outGrad, ga);
                    } else {
                        for (int i = 0; i < ga.data.length; i++) {
                            float o = out.data[i]; // tanh output
                            ga.data[i] = outGrad.data[i] * (1 - o * o);
                        }
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor leaky_relu(Tensor a, float negativeSlope) {
        Tensor out = new Tensor(a.shape);
        if (a.isGPU()) {
            out.toGPU();
            CUDAOps.leakyReluForward(a, out, negativeSlope);
        } else {
            for (int i = 0; i < a.data.length; i++) {
                float v = a.data[i];
                out.data[i] = v > 0 ? v : v * negativeSlope;
            }
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    if (a.isGPU()) {
                        ga.toGPU();
                        CUDAOps.leakyReluBackward(a, outGrad, ga, negativeSlope);
                    } else {
                        for (int i = 0; i < ga.data.length; i++) {
                            ga.data[i] = (a.data[i] > 0 ? 1f : negativeSlope) * outGrad.data[i];
                        }
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    // convert between NN.Mat (2D) and Tensor
    public static Tensor fromMat(NN.Mat m) {
        return new Tensor(m.es.clone(), m.rows, m.cols);
    }

    public static NN.Mat toMat(Tensor t) {
        if (t.shape.length != 2)
            throw new IllegalArgumentException("toMat requires 2D tensor");
        NN.Mat m = NN.mat_alloc(t.shape[0], t.shape[1]);
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
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.sin(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor cos(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.cos(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor tan(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.tan(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor exp(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.exp(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor log(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.log(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor ceil(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.ceil(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor floor(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.floor(a.data[i]);
        if (a.isGPU()) out.toGPU();
        return out;
    }

    public static Tensor round(Tensor a) {
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.round(a.data[i]);
        if (a.isGPU()) out.toGPU();
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
        a.toCPU();
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++)
            out.data[i] = (float) Math.pow(a.data[i], exp);
        if (a.isGPU()) out.toGPU();
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
    
    // --- Batch 1 Activations & Softmax ---

    public static Tensor softmax(Tensor a, int dim) {
        a.toCPU();
        if (dim < 0)
            dim += a.shape.length;
        int nd = a.shape.length;
        int dimSize = a.shape[dim];
        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= a.shape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < nd; i++)
            innerSize *= a.shape[i];

        Tensor out = new Tensor(a.shape);
        if (a.isGPU()) out.toGPU();
        for (int i = 0; i < outerSize; i++) {
            for (int k = 0; k < innerSize; k++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < dimSize; j++) {
                    float v = a.data[i * dimSize * innerSize + j * innerSize + k];
                    if (v > maxVal)
                        maxVal = v;
                }
                double sum = 0;
                for (int j = 0; j < dimSize; j++) {
                    float v = a.data[i * dimSize * innerSize + j * innerSize + k];
                    float ev = (float) Math.exp(v - maxVal);
                    out.data[i * dimSize * innerSize + j * innerSize + k] = ev;
                    sum += ev;
                }
                for (int j = 0; j < dimSize; j++) {
                    out.data[i * dimSize * innerSize + j * innerSize + k] /= (float) sum;
                }
            }
        }

        if (is_grad_enabled() && a.requires_grad) {
            final int fOuterSize = outerSize;
            final int fInnerSize = innerSize;
            final int fDimSize = dimSize;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < fOuterSize; i++) {
                        for (int k = 0; k < fInnerSize; k++) {
                            float dot = 0;
                            for (int j = 0; j < fDimSize; j++) {
                                int idx = i * (fDimSize * fInnerSize) + j * fInnerSize + k;
                                dot += outGrad.data[idx] * out.data[idx];
                            }
                            for (int j = 0; j < fDimSize; j++) {
                                int idx = i * (fDimSize * fInnerSize) + j * fInnerSize + k;
                                ga.data[idx] = out.data[idx] * (outGrad.data[idx] - dot);
                            }
                        }
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor log_softmax(Tensor a, int dim) {
        a.toCPU();
        if (dim < 0)
            dim += a.shape.length;
        int nd = a.shape.length;
        int dimSize = a.shape[dim];
        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= a.shape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < nd; i++)
            innerSize *= a.shape[i];

        Tensor out = new Tensor(a.shape);
        if (a.isGPU()) out.toGPU();
        for (int i = 0; i < outerSize; i++) {
            for (int k = 0; k < innerSize; k++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < dimSize; j++) {
                    float v = a.data[i * dimSize * innerSize + j * innerSize + k];
                    if (v > maxVal)
                        maxVal = v;
                }
                double sum = 0;
                for (int j = 0; j < dimSize; j++) {
                    float v = a.data[i * dimSize * innerSize + j * innerSize + k];
                    sum += Math.exp(v - maxVal);
                }
                float logSum = (float) (Math.log(sum) + maxVal);
                for (int j = 0; j < dimSize; j++) {
                    float v = a.data[i * dimSize * innerSize + j * innerSize + k];
                    out.data[i * dimSize * innerSize + j * innerSize + k] = v - logSum;
                }
            }
        }

        if (is_grad_enabled() && a.requires_grad) {
            final int fDim = dim;
            final int fOuterSize = outerSize;
            final int fInnerSize = innerSize;
            final int fDimSize = dimSize;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    Tensor soft = softmax(a, fDim);
                    for (int i = 0; i < fOuterSize; i++) {
                        for (int k = 0; k < fInnerSize; k++) {
                            float sumGrad = 0;
                            for (int j = 0; j < fDimSize; j++) {
                                int idx = i * (fDimSize * fInnerSize) + j * fInnerSize + k;
                                sumGrad += outGrad.data[idx];
                            }
                            for (int j = 0; j < fDimSize; j++) {
                                int idx = i * (fDimSize * fInnerSize) + j * fInnerSize + k;
                                ga.data[idx] = outGrad.data[idx] - soft.data[idx] * sumGrad;
                            }
                        }
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor cosine_similarity(Tensor x1, Tensor x2, int dim, float eps) {
        if (dim < 0)
            dim += x1.shape.length;
        int nd = x1.shape.length;
        int dimSize = x1.shape[dim];
        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= x1.shape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < nd; i++)
            innerSize *= x1.shape[i];

        int[] outShape;
        if (nd == 1) {
            outShape = new int[] { 1 };
        } else {
            outShape = new int[nd - 1];
            int p = 0;
            for (int i = 0; i < nd; i++) {
                if (i != dim)
                    outShape[p++] = x1.shape[i];
            }
        }

        Tensor res = new Tensor(outShape);
        float[] norm1 = new float[outerSize * innerSize];
        float[] norm2 = new float[outerSize * innerSize];

        for (int i = 0; i < outerSize; i++) {
            for (int k = 0; k < innerSize; k++) {
                double dot = 0;
                double n1 = 0;
                double n2 = 0;
                for (int j = 0; j < dimSize; j++) {
                    int idx = i * dimSize * innerSize + j * innerSize + k;
                    float v1 = x1.data[idx];
                    float v2 = x2.data[idx];
                    dot += v1 * v2;
                    n1 += v1 * v1;
                    n2 += v2 * v2;
                }
                float sn1 = (float) Math.sqrt(n1);
                float sn2 = (float) Math.sqrt(n2);
                float den = Math.max(sn1 * sn2, eps);
                res.data[i * innerSize + k] = (float) (dot / den);
                norm1[i * innerSize + k] = sn1;
                norm2[i * innerSize + k] = sn2;
            }
        }

        if (is_grad_enabled() && (x1.requires_grad || x2.requires_grad)) {
            final int fOuterSize = outerSize;
            final int fInnerSize = innerSize;
            final int fDimSize = dimSize;
            final float fEps = eps;
            res.requires_grad = true;
            res.grad_fn = new Tensor.GradFn(x1, x2) {
                public void apply(Tensor outGrad) {
                    if (x1.requires_grad) {
                        Tensor g1 = new Tensor(x1.shape);
                        for (int i = 0; i < fOuterSize; i++) {
                            for (int k = 0; k < fInnerSize; k++) {
                                int outIdx = i * fInnerSize + k;
                                float og = outGrad.data[outIdx];
                                float sim = res.data[outIdx];
                                float n1 = norm1[outIdx];
                                float n2 = norm2[outIdx];
                                float den = Math.max(n1 * n2, fEps);
                                for (int j = 0; j < fDimSize; j++) {
                                    int idx = i * fDimSize * fInnerSize + j * fInnerSize + k;
                                    g1.data[idx] = og * ((x2.data[idx] / den) - (sim * x1.data[idx] / (n1 * n1 + fEps)));
                                }
                            }
                        }
                        x1.backwardStep(g1);
                    }
                    if (x2.requires_grad) {
                        Tensor g2 = new Tensor(x2.shape);
                        for (int i = 0; i < fOuterSize; i++) {
                            for (int k = 0; k < fInnerSize; k++) {
                                int outIdx = i * fInnerSize + k;
                                float og = outGrad.data[outIdx];
                                float sim = res.data[outIdx];
                                float n1 = norm1[outIdx];
                                float n2 = norm2[outIdx];
                                float den = Math.max(n1 * n2, fEps);
                                for (int j = 0; j < fDimSize; j++) {
                                    int idx = i * fDimSize * fInnerSize + j * fInnerSize + k;
                                    g2.data[idx] = og * ((x1.data[idx] / den) - (sim * x2.data[idx] / (n2 * n2 + fEps)));
                                }
                            }
                        }
                        x2.backwardStep(g2);
                    }
                }
            };
        }
        return res;
    }

    public static Tensor pairwise_distance(Tensor x1, Tensor x2, float p, float eps) {
        int dim = x1.shape.length - 1;
        if (dim < 0)
            dim = 0;
        int nd = x1.shape.length;
        int dimSize = x1.shape[dim];
        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= x1.shape[i];

        int[] outShape;
        if (nd == 1) {
            outShape = new int[] { 1 };
        } else {
            outShape = new int[nd - 1];
            for (int i = 0; i < nd - 1; i++)
                outShape[i] = x1.shape[i];
        }

        Tensor out = new Tensor(outShape);

        for (int i = 0; i < outerSize; i++) {
            double sumValue = 0;
            for (int j = 0; j < dimSize; j++) {
                float diff = x1.data[i * dimSize + j] - x2.data[i * dimSize + j];
                sumValue += Math.pow(Math.abs(diff), p);
            }
            out.data[i] = (float) Math.pow(sumValue + eps, 1.0 / p);
        }

        if (is_grad_enabled() && (x1.requires_grad || x2.requires_grad)) {
            final int fOuterSize = outerSize;
            final int fDimSize = dimSize;
            final float fP = p;
            final float fEps = eps;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x1, x2) {
                public void apply(Tensor outGrad) {
                    if (x1.requires_grad) {
                        Tensor g1 = new Tensor(x1.shape);
                        for (int i = 0; i < fOuterSize; i++) {
                            float og = outGrad.data[i];
                            float dist = out.data[i];
                            for (int j = 0; j < fDimSize; j++) {
                                float diff = x1.data[i * fDimSize + j] - x2.data[i * fDimSize + j];
                                g1.data[i * fDimSize + j] = (float) (og * Math.signum(diff)
                                        * Math.pow(Math.abs(diff), fP - 1) / Math.pow(dist, fP - 1 + 1e-12));
                            }
                        }
                        x1.backwardStep(g1);
                    }
                    if (x2.requires_grad) {
                        Tensor g2 = new Tensor(x2.shape);
                        for (int i = 0; i < fOuterSize; i++) {
                            float og = outGrad.data[i];
                            float dist = out.data[i];
                            for (int j = 0; j < fDimSize; j++) {
                                float diff = x1.data[i * fDimSize + j] - x2.data[i * fDimSize + j];
                                g2.data[i * fDimSize + j] = (float) (-og * Math.signum(diff)
                                        * Math.pow(Math.abs(diff), fP - 1) / Math.pow(dist, fP - 1 + 1e-12));
                            }
                        }
                        x2.backwardStep(g2);
                    }
                }
            };
        }
        return out;
    }

    public static Tensor gelu(Tensor a) {
        // approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++) {
            float x = a.data[i];
            float inner = (float) (Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)));
            out.data[i] = (float) (0.5 * x * (1.0 + Math.tanh(inner)));
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    final float sqrt2Pi = (float) Math.sqrt(2.0 / Math.PI);
                    for (int i = 0; i < ga.data.length; i++) {
                        float x = a.data[i];
                        float x3 = x * x * x;
                        float inner = sqrt2Pi * (x + 0.044715f * x3);
                        float th = (float) Math.tanh(inner);
                        float sech2 = 1.0f - th * th;
                        float deriv = 0.5f * (1.0f + th) + 0.5f * x * sech2 * sqrt2Pi * (1.0f + 3.0f * 0.044715f * x * x);
                        ga.data[i] = outGrad.data[i] * deriv;
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor elu(Tensor a, float alpha) {
        Tensor out = new Tensor(a.shape);
        for (int i = 0; i < a.data.length; i++) {
            float x = a.data[i];
            out.data[i] = x > 0 ? x : (float) (alpha * (Math.exp(x) - 1.0));
        }

        if (is_grad_enabled() && a.requires_grad) {
            final float fAlpha = alpha;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++) {
                        float x = a.data[i];
                        float deriv = x > 0 ? 1f : (float) (fAlpha * Math.exp(x));
                        ga.data[i] = outGrad.data[i] * deriv;
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor silu(Tensor a) {
        // x * sigmoid(x)
        Tensor out = new Tensor(a.shape);
        Tensor sig = sigmoid(a);
        for (int i = 0; i < a.data.length; i++) {
            out.data[i] = a.data[i] * sig.data[i];
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++) {
                        float s = sig.data[i];
                        float x = a.data[i];
                        // d(x*sig(x))/dx = sig(x) + x * sig(x)*(1-sig(x))
                        float deriv = s + x * s * (1f - s);
                        ga.data[i] = outGrad.data[i] * deriv;
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
    }

    public static Tensor dropout(Tensor a, float p, boolean training) {
        if (p < 0f || p >= 1f)
            throw new IllegalArgumentException("dropout: p must be in [0, 1)");
        if (!training || p == 0f)
            return a;

        Tensor out = new Tensor(a.shape);
        float scale = 1f / (1f - p);
        final float[] mask = new float[a.data.length];

        for (int i = 0; i < a.data.length; i++) {
            if (globalR.nextFloat() >= p) {
                mask[i] = 1f;
                out.data[i] = a.data[i] * scale;
            } else {
                mask[i] = 0f;
                out.data[i] = 0f;
            }
        }

        if (is_grad_enabled() && a.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(a) {
                public void apply(Tensor outGrad) {
                    Tensor ga = new Tensor(a.shape);
                    for (int i = 0; i < ga.data.length; i++) {
                        ga.data[i] = outGrad.data[i] * mask[i] * scale;
                    }
                    a.backwardStep(ga);
                }
            };
        }
        return out;
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


    public static Tensor one_hot(Tensor indices, int numClasses) {
        // indices: [N] or [N, L]
        int n = indices.numel();
        int[] outShape = new int[indices.shape.length + 1];
        System.arraycopy(indices.shape, 0, outShape, 0, indices.shape.length);
        outShape[outShape.length - 1] = numClasses;

        Tensor out = new Tensor(outShape);
        for (int i = 0; i < n; i++) {
            int classIdx = (int) indices.data[i];
            if (classIdx >= 0 && classIdx < numClasses) {
                out.data[i * numClasses + classIdx] = 1.0f;
            }
        }
        return out;
    }

    // --- Batch 4: Conv1d, Bilinear, OneHot ---

    public static Tensor conv1d(Tensor x, Tensor weight, Tensor bias, int stride, int padding) {
        final int nd = x.shape.length;
        if (nd < 2)
            throw new IllegalArgumentException("Expected x to have at least 2 dims");
        final int batch = (nd == 2) ? x.shape[0] : (nd == 3 ? x.shape[0] : 1);
        final int inC = (nd == 2) ? weight.shape[0] / (weight.numel() / (weight.shape[weight.shape.length - 1] * weight.shape[0])) : x.shape[nd - 2];
        // Wait, weight shape for 1D should be [outC, inC*kW] or [outC, inC, kW]
        // Let's assume weight is [outC, inC, kW] for clarity.
        final int outC = weight.shape[0];
        final int weightInC = weight.shape[1];
        final int kW = weight.shape[2];
        final int inL = (nd == 2) ? (x.shape[1] / weightInC) : x.shape[nd - 1];
        final int outL = (inL + 2 * padding - kW) / stride + 1;

        int[] outShape = (nd == 3) ? new int[] { batch, outC, outL } : new int[] { batch, outC * outL };
        Tensor out = new Tensor(outShape);

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outC; oc++) {
                for (int ol = 0; ol < outL; ol++) {
                    float sum = 0f;
                    for (int ic = 0; ic < weightInC; ic++) {
                        for (int k = 0; k < kW; k++) {
                            int il = ol * stride - padding + k;
                            if (il >= 0 && il < inL) {
                                float valX = x.data[b * (weightInC * inL) + ic * inL + il];
                                float valW = weight.data[oc * (weightInC * kW) + ic * kW + k];
                                sum += valX * valW;
                            }
                        }
                    }
                    if (bias != null)
                        sum += bias.data[oc];
                    out.data[b * (outC * outL) + oc * outL + ol] = sum;
                }
            }
        }

        if (is_grad_enabled() && (x.requires_grad || weight.requires_grad || (bias != null && bias.requires_grad))) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x, weight, bias) {
                public void apply(Tensor outGrad) {
                    if (x.requires_grad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                for (int ol = 0; ol < outL; ol++) {
                                    float og = outGrad.data[b * (outC * outL) + oc * outL + ol];
                                    for (int ic = 0; ic < weightInC; ic++) {
                                        for (int k = 0; k < kW; k++) {
                                            int il = ol * stride - padding + k;
                                            if (il >= 0 && il < inL) {
                                                gx.data[b * (weightInC * inL) + ic * inL + il] += og * weight.data[oc * (weightInC * kW) + ic * kW + k];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        x.backwardStep(gx);
                    }
                    if (weight.requires_grad) {
                        Tensor gw = new Tensor(weight.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                for (int ol = 0; ol < outL; ol++) {
                                    float og = outGrad.data[b * (outC * outL) + oc * outL + ol];
                                    for (int ic = 0; ic < weightInC; ic++) {
                                        for (int k = 0; k < kW; k++) {
                                            int il = ol * stride - padding + k;
                                            if (il >= 0 && il < inL) {
                                                gw.data[oc * (weightInC * kW) + ic * kW + k] += og * x.data[b * (weightInC * inL) + ic * inL + il];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        weight.backwardStep(gw);
                    }
                    if (bias != null && bias.requires_grad) {
                        Tensor gb = new Tensor(bias.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                for (int ol = 0; ol < outL; ol++) {
                                    gb.data[oc] += outGrad.data[b * (outC * outL) + oc * outL + ol];
                                }
                            }
                        }
                        bias.backwardStep(gb);
                    }
                }
            };
        }
        return out;
    }

    public static Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, Tensor bias) {
        // x1: [N, D1], x2: [N, D2], weight: [Out, D1, D2], bias: [Out]
        int batch = x1.shape[0];
        int outC = weight.shape[0];
        int d1 = weight.shape[1];
        int d2 = weight.shape[2];

        Tensor out = new Tensor(batch, outC);

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outC; oc++) {
                float sum = 0f;
                for (int i = 0; i < d1; i++) {
                    for (int j = 0; j < d2; j++) {
                        sum += x1.data[b * d1 + i] * weight.data[oc * d1 * d2 + i * d2 + j] * x2.data[b * d2 + j];
                    }
                }
                if (bias != null)
                    sum += bias.data[oc];
                out.data[b * outC + oc] = sum;
            }
        }

        if (is_grad_enabled() && (x1.requires_grad || x2.requires_grad || weight.requires_grad || (bias != null && bias.requires_grad))) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x1, x2, weight, bias) {
                public void apply(Tensor outGrad) {
                    if (x1.requires_grad) {
                        Tensor g1 = new Tensor(x1.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                float og = outGrad.data[b * outC + oc];
                                for (int i = 0; i < d1; i++) {
                                    for (int j = 0; j < d2; j++) {
                                        g1.data[b * d1 + i] += og * weight.data[oc * d1 * d2 + i * d2 + j] * x2.data[b * d2 + j];
                                    }
                                }
                            }
                        }
                        x1.backwardStep(g1);
                    }
                    if (x2.requires_grad) {
                        Tensor g2 = new Tensor(x2.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                float og = outGrad.data[b * outC + oc];
                                for (int i = 0; i < d1; i++) {
                                    for (int j = 0; j < d2; j++) {
                                        g2.data[b * d2 + j] += og * x1.data[b * d1 + i] * weight.data[oc * d1 * d2 + i * d2 + j];
                                    }
                                }
                            }
                        }
                        x2.backwardStep(g2);
                    }
                    if (weight.requires_grad) {
                        Tensor gw = new Tensor(weight.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                float og = outGrad.data[b * outC + oc];
                                for (int i = 0; i < d1; i++) {
                                    for (int j = 0; j < d2; j++) {
                                        gw.data[oc * d1 * d2 + i * d2 + j] += og * x1.data[b * d1 + i] * x2.data[b * d2 + j];
                                    }
                                }
                            }
                        }
                        weight.backwardStep(gw);
                    }
                    if (bias != null && bias.requires_grad) {
                        Tensor gb = new Tensor(bias.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int oc = 0; oc < outC; oc++) {
                                gb.data[oc] += outGrad.data[b * outC + oc];
                            }
                        }
                        bias.backwardStep(gb);
                    }
                }
            };
        }
        return out;
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

    public static Tensor bmm_TEMP(Tensor a, Tensor b) {
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
        boolean anyGPU = false;
        for (Tensor t : tensors) {
            totalDimSize += t.shape[dim];
            if (t.isGPU()) anyGPU = true;
        }

        int[] newShape = baseShape.clone();
        newShape[dim] = totalDimSize;
        Tensor res = new Tensor(newShape);
        if (anyGPU) res.toGPU();

        if (anyGPU) {
            // Ensure all are on GPU for the fast operation
            for (Tensor t : tensors) t.toGPU();
            CUDAOps.concat(tensors, res, dim);
        } else {
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
        if (input.isGPU()) out.toGPU();

        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= input.shape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < input.shape.length; i++)
            innerSize *= input.shape[i];

        final int finalDim = dim;
        final int finalStart = start;
        final int finalLength = length;
        final int finalInnerSize = innerSize;
        final int finalOuterSize = outerSize;
        final int finalOldDimSize = input.shape[dim];

        if (input.isGPU()) {
            CUDAOps.narrow(input, out, dim, start, length);
        } else {
            for (int i = 0; i < outerSize; i++) {
                // Copy a block for each outer index
                System.arraycopy(
                        input.data, (i * finalOldDimSize + finalStart) * finalInnerSize,
                        out.data, (i * finalLength) * finalInnerSize,
                        finalLength * finalInnerSize);
            }
        }

        if (is_grad_enabled() && input.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(input) {
                public void apply(Tensor outGrad) {
                    Tensor grad = new Tensor(input.shape);
                    if (outGrad.isGPU()) {
                        grad.toGPU();
                        // Zero grad on GPU
                        CUDAOps.mul(grad, 0.0f, grad); // Simple way to zero
                        // Put back gradients into the right place
                        // We need a way to add narrowed grad back. 
                        // For now, let's keep sub-optimal copy or optimize later.
                        // Actually, narrow backward is basically putting a smaller tensor into a larger one.
                        // We can use a custom "scatter" or "assign_narrow" kernel.
                        // For now, use CPU fallback to be safe.
                        outGrad.toCPU();
                    }
                    
                    for (int i = 0; i < finalOuterSize; i++) {
                        System.arraycopy(
                                outGrad.data, (i * finalLength) * finalInnerSize,
                                grad.data, (i * finalOldDimSize + finalStart) * finalInnerSize,
                                finalLength * finalInnerSize);
                    }
                    if (input.isGPU()) grad.toGPU();
                    input.backwardStep(grad);
                }
            };
        }
        return out;
    }

    public static Tensor embedding(Tensor weight, Tensor indices) {
        // indices shape [...]
        // weight shape [num_embeddings, embedding_dim]
        int[] inS = indices.shape;
        int d = weight.shape[1];
        int n = indices.numel();
        int[] outS = new int[inS.length + 1];
        System.arraycopy(inS, 0, outS, 0, inS.length);
        outS[inS.length] = d;
        Tensor out = new Tensor(outS);

        boolean useGPU = weight.isGPU() && CUDAOps.isAvailable();
        if (useGPU) {
            indices.toGPU();
            out.toGPU();
            CUDAOps.embeddingForward(weight, indices, out);
        } else {
            weight.toCPU();
            indices.toCPU();
            for (int i = 0; i < n; i++) {
                int idx = (int) indices.data[i];
                System.arraycopy(weight.data, idx * d, out.data, i * d, d);
            }
            if (weight.isGPU() || indices.isGPU()) out.toGPU();
        }
        if (is_grad_enabled() && weight.requires_grad) {
            out.requires_grad = true;
            final boolean gpuBackward = useGPU;
            out.grad_fn = new Tensor.GradFn(weight) {
                public void apply(Tensor outGrad) {
                    Tensor gw = new Tensor(weight.shape);
                    if (gpuBackward) {
                        gw.toGPU();
                        // gw is zeroed on CPU, push zeros to GPU
                        if (!outGrad.isGPU()) outGrad.toGPU();
                        CUDAOps.embeddingBackward(gw, indices, outGrad);
                    } else {
                        outGrad.toCPU();
                        for (int i = 0; i < n; i++) {
                            int idx = (int) indices.data[i];
                            for (int j = 0; j < d; j++) {
                                gw.data[idx * d + j] += outGrad.data[i * d + j];
                            }
                        }
                    }
                    weight.backwardStep(gw);
                }
            };
        }
        return out;
    }

    // ---- Torch.nn.init.* weight initialization API ----
    public static class nn {
        public static class init {
            
            /** Compute fan_in and fan_out from tensor shape.
             *  2D [out, in]: fan_in=in, fan_out=out
             *  4D [out, in, kH, kW]: fan_in=in*kH*kW, fan_out=out*kH*kW
             *  1D [n]: fan_in=n, fan_out=n
             */
            public static int[] calculateFanInOut(Tensor t) {
                int[] s = t.shape;
                if (s.length == 1) {
                    return new int[]{s[0], s[0]};
                } else if (s.length == 2) {
                    return new int[]{s[1], s[0]};
                } else {
                    // Conv: [outC, inC, k1, k2, ...]
                    int receptiveField = 1;
                    for (int i = 2; i < s.length; i++) receptiveField *= s[i];
                    return new int[]{s[1] * receptiveField, s[0] * receptiveField};
                }
            }

            /** Gain for activation functions. */
            public static float calculateGain(String activation) {
                switch (activation) {
                    case "linear": case "sigmoid": return 1.0f;
                    case "tanh": return 5.0f / 3.0f;
                    case "relu": return (float) Math.sqrt(2.0);
                    default: return 1.0f;
                }
            }

            public static float calculateGain(String activation, float param) {
                if ("leaky_relu".equals(activation)) {
                    return (float) Math.sqrt(2.0 / (1.0 + param * param));
                }
                return calculateGain(activation);
            }

            /** Fill tensor with uniform random in [a, b). */
            public static void uniform_(Tensor t, float a, float b) {
                t.toCPU();
                float range = b - a;
                for (int i = 0; i < t.data.length; i++)
                    t.data[i] = a + globalR.nextFloat() * range;
                t.markDirtyOnCPU();
            }

            /** Fill tensor with normal(mean, std). */
            public static void normal_(Tensor t, float mean, float std) {
                t.toCPU();
                for (int i = 0; i < t.data.length; i++)
                    t.data[i] = mean + (float) nextGaussian(globalR) * std;
                t.markDirtyOnCPU();
            }

            /** Fill with zeros. */
            public static void zeros_(Tensor t) {
                t.toCPU();
                java.util.Arrays.fill(t.data, 0f);
                t.markDirtyOnCPU();
            }

            /** Fill with ones. */
            public static void ones_(Tensor t) {
                t.toCPU();
                java.util.Arrays.fill(t.data, 1f);
                t.markDirtyOnCPU();
            }

            /** Fill with constant value. */
            public static void constant_(Tensor t, float val) {
                t.toCPU();
                java.util.Arrays.fill(t.data, val);
                t.markDirtyOnCPU();
            }

            /** Xavier (Glorot) uniform: U[-a, a] where a = gain * sqrt(6/(fan_in+fan_out)). */
            public static void xavier_uniform_(Tensor t) {
                xavier_uniform_(t, 1.0f);
            }

            public static void xavier_uniform_(Tensor t, float gain) {
                int[] fan = calculateFanInOut(t);
                float a = gain * (float) Math.sqrt(6.0 / (fan[0] + fan[1]));
                uniform_(t, -a, a);
            }

            /** Xavier (Glorot) normal: N(0, std) where std = gain * sqrt(2/(fan_in+fan_out)). */
            public static void xavier_normal_(Tensor t) {
                xavier_normal_(t, 1.0f);
            }

            public static void xavier_normal_(Tensor t, float gain) {
                int[] fan = calculateFanInOut(t);
                float std = gain * (float) Math.sqrt(2.0 / (fan[0] + fan[1]));
                normal_(t, 0f, std);
            }

            /** Kaiming (He) uniform: U[-bound, bound] where bound = gain * sqrt(3/fan_in). */
            public static void kaiming_uniform_(Tensor t) {
                kaiming_uniform_(t, calculateGain("relu"));
            }

            public static void kaiming_uniform_(Tensor t, float gain) {
                int[] fan = calculateFanInOut(t);
                float std = gain / (float) Math.sqrt(fan[0]);
                float bound = (float) Math.sqrt(3.0) * std;
                uniform_(t, -bound, bound);
            }

            /** Kaiming (He) normal: N(0, std) where std = gain / sqrt(fan_in). */
            public static void kaiming_normal_(Tensor t) {
                kaiming_normal_(t, calculateGain("relu"));
            }

            public static void kaiming_normal_(Tensor t, float gain) {
                int[] fan = calculateFanInOut(t);
                float std = gain / (float) Math.sqrt(fan[0]);
                normal_(t, 0f, std);
            }
        }
    }
}
