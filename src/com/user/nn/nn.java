package com.user.nn;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public class nn {
    public static class Mat {
        public int rows;
        public int cols;
        public float[] es;
    }

    public final float MAT_AT(Mat m, int i, int j) {
        return m.es[i * m.cols + j];
    }

    public Mat mat_alloc(int rows, int cols) {
        Mat m = new Mat();
        m.rows = rows;
        m.cols = cols;
        m.es = new float[rows * cols];
        return m;
    }

    public void mat_dot(Mat dst, Mat a, Mat b) {
        // Use Tensor-based matmul for correctness and reuse
        if (a.cols != b.rows) throw new IllegalArgumentException("Incompatible matrix dimensions for dot: a.cols must equal b.rows");
        if (dst.rows != a.rows || dst.cols != b.cols) throw new IllegalArgumentException("Destination matrix has wrong dimensions");
        Tensor ta = Torch.fromMat(a);
        Tensor tb = Torch.fromMat(b);
        Tensor tc = Torch.matmul(ta, tb);
        // copy back
        if (tc.shape.length != 2 || tc.shape[0] != dst.rows || tc.shape[1] != dst.cols) throw new IllegalStateException("matmul result shape mismatch");
        System.arraycopy(tc.data, 0, dst.es, 0, dst.es.length);
    }

    public void mat_sum(Mat dst, Mat a) {
        if (dst.rows != a.rows || dst.cols != a.cols) throw new IllegalArgumentException("Matrices must have the same dimensions for addition");
        Tensor ta = Torch.fromMat(dst); Tensor tb = Torch.fromMat(a);
        Tensor tr = Torch.add(ta, tb);
        System.arraycopy(tr.data, 0, dst.es, 0, dst.es.length);
    }

    public void mat_sub(Mat dst, Mat a) {
        if (dst.rows != a.rows || dst.cols != a.cols) throw new IllegalArgumentException("Matrices must have the same dimensions for subtraction");
        Tensor ta = Torch.fromMat(dst); Tensor tb = Torch.fromMat(a);
        Tensor tr = Torch.sub(ta, tb);
        System.arraycopy(tr.data, 0, dst.es, 0, dst.es.length);
    }

    public void mat_print(Mat m) {
        Tensor t = Torch.fromMat(m);
        System.out.println(t.toString());
    }

    public void mat_rand(Mat m, float min, float max) {
        Tensor t = Torch.rand(new int[]{m.rows, m.cols});
        // scale to [min,max)
        float range = max - min;
        for (int i=0;i<t.data.length;i++) t.data[i] = min + t.data[i] * range;
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    } 

    public void mat_fill(Mat m, float value) {
        Tensor t = Torch.full(new int[]{m.rows, m.cols}, value);
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // Elementwise apply (in-place) using a lambda-like helper interface
    public interface ElemOp {
        float apply(float x);
    }

    public void mat_apply_inplace(Mat m, ElemOp op) {
        // apply elementwise via Tensor
        Tensor t = Torch.fromMat(m);
        for (int i=0;i<t.data.length;i++) t.data[i] = op.apply(t.data[i]);
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // Deterministic random fill
    public void mat_rand_seed(Mat m, long seed, float min, float max) {
        // deterministic rand via Torch.randn seeded by seed
        Torch.manual_seed(seed);
        Tensor t = Torch.rand(new int[]{m.rows, m.cols});
        float range = max - min;
        for (int i=0;i<t.data.length;i++) t.data[i] = min + t.data[i] * range;
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // CSV read/write utilities (rows lines, comma-separated)
    public void writeMatCSV(Mat m, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                sb.append(m.es[i * m.cols + j]);
                if (j + 1 < m.cols) sb.append(',');
            }
            sb.append('\n');
        }
        Files.createDirectories(Paths.get(path).getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(path))) {
            bw.write(sb.toString());
        }
    }

    public Mat readMatCSV(String path) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(path));
        if (lines.size() == 0) return null;
        int rows = lines.size();
        String[] first = lines.get(0).split(",");
        int cols = first.length;
        Mat m = mat_alloc(rows, cols);
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).trim().split(",");
            for (int j = 0; j < cols; j++) {
                m.es[i * cols + j] = Float.parseFloat(parts[j]);
            }
        }
        return m;
    }

    // --- NN core classes (lightweight, non-backprop proof) ---
    public static class Parameter {
        public Mat data;
        public boolean requiresGrad = true;

        public Parameter(Mat data) {
            this.data = data;
        }
    }

    public static abstract class Module {
        protected Map<String, Module> children = new LinkedHashMap<>();
        protected Map<String, Parameter> params = new LinkedHashMap<>();

        public void addModule(String name, Module m) {
            children.put(name, m);
        }

        public void addParameter(String name, Parameter p) {
            params.put(name, p);
        }

        public Module getModule(String name) {
            return children.get(name);
        }

        public Parameter getParameter(String name) {
            return params.get(name);
        }

        public List<Parameter> parameters() {
            List<Parameter> out = new ArrayList<>();
            out.addAll(params.values());
            for (Module m : children.values()) {
                out.addAll(m.parameters());
            }
            return out;
        }

        public List<Module> modules() {
            List<Module> out = new ArrayList<>();
            out.addAll(children.values());
            return out;
        }

        public abstract Mat forward(Mat x);

        public Mat apply(Mat x) {
            return forward(x);
        }
    }

    public static class Sequential extends Module {
        private final List<Module> list = new ArrayList<>();

        public Sequential() {}

        public void add(Module m) {
            String name = "" + list.size();
            list.add(m);
            addModule(name, m);
        }

        @Override
        public Mat forward(Mat x) {
            Mat out = x;
            for (Module m : list) {
                out = m.forward(out);
            }
            return out;
        }
    }

    public static class ModuleList extends Module {
        private final List<Module> list = new ArrayList<>();

        public void add(Module m) {
            String name = "" + list.size();
            list.add(m);
            addModule(name, m);
        }

        public Module get(int idx) { return list.get(idx); }

        @Override
        public Mat forward(Mat x) {
            throw new UnsupportedOperationException("ModuleList does not implement forward directly");
        }
    }

    public static class ModuleDict extends Module {
        public void put(String name, Module m) {
            addModule(name, m);
        }

        @Override
        public Mat forward(Mat x) {
            throw new UnsupportedOperationException("ModuleDict does not implement forward directly");
        }
    }

    // --- Layers ---
    public static class Linear extends Module {
        public int inFeatures;
        public int outFeatures;
        public Parameter weight; // shape: inFeatures x outFeatures
        public Parameter bias;   // shape: 1 x outFeatures (row vector)

        public Linear(nn outer, int inFeatures, int outFeatures, boolean useBias) {
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            Mat w = outer.mat_alloc(inFeatures, outFeatures);
            outer.mat_rand(w, -0.08f, 0.08f);
            this.weight = new Parameter(w);
            addParameter("weight", this.weight);
            if (useBias) {
                Mat b = outer.mat_alloc(1, outFeatures);
                outer.mat_fill(b, 0.0f);
                this.bias = new Parameter(b);
                addParameter("bias", this.bias);
            }
        }

        @Override
        public Mat forward(Mat input) {
            if (input.cols != inFeatures) {
                throw new IllegalArgumentException("Input features mismatch: expected " + inFeatures + " got " + input.cols);
            }
            Mat out = new Mat();
            out.rows = input.rows;
            out.cols = outFeatures;
            out.es = new float[out.rows * out.cols];
            // out = input (batch x in) dot weight (in x out) -> (batch x out)
            // reuse outer mat_dot via a temporary nn instance is not necessary; implement simple dot here
            for (int i = 0; i < out.rows; i++) {
                for (int j = 0; j < out.cols; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < inFeatures; k++) {
                        sum += input.es[i * input.cols + k] * weight.data.es[k * weight.data.cols + j];
                    }
                    out.es[i * out.cols + j] = sum;
                }
            }
            if (bias != null) {
                for (int i = 0; i < out.rows; i++) {
                    for (int j = 0; j < out.cols; j++) {
                        out.es[i * out.cols + j] += bias.data.es[j];
                    }
                }
            }
            return out;
        }
    }

    public static class ReLU extends Module {
        @Override
        public Mat forward(Mat x) {
            Mat out = new Mat();
            out.rows = x.rows;
            out.cols = x.cols;
            out.es = new float[out.rows * out.cols];
            int n = out.rows * out.cols;
            for (int i = 0; i < n; i++) {
                float v = x.es[i];
                out.es[i] = v > 0 ? v : 0f;
            }
            return out;
        }
    }

    // Functional utilities
    public static class F {
        public static Mat relu(nn outer, Mat x) {
            ReLU r = new ReLU();
            return r.forward(x);
        }
        public static float mse_loss(Mat pred, Mat target) {
            if (pred.rows != target.rows || pred.cols != target.cols) {
                throw new IllegalArgumentException("mse_loss: shape mismatch");
            }
            int n = pred.rows * pred.cols;
            float s = 0f;
            for (int i = 0; i < n; i++) {
                float d = pred.es[i] - target.es[i];
                s += d * d;
            }
            return s / n;
        }

        public static float cross_entropy_logits(Mat logits, int[] targets) {
            // logits: batch x classes; targets: length batch with class indices
            if (logits.rows != targets.length) throw new IllegalArgumentException("cross_entropy: batch size mismatch");
            int batch = logits.rows;
            int classes = logits.cols;
            float total = 0f;
            for (int i = 0; i < batch; i++) {
                // find max for numerical stability
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < classes; j++) if (logits.es[i * classes + j] > max) max = logits.es[i * classes + j];
                double sum = 0.0;
                for (int j = 0; j < classes; j++) sum += Math.exp(logits.es[i * classes + j] - max);
                double logsum = Math.log(sum) + max;
                int t = targets[i];
                double logit_target = logits.es[i * classes + t];
                total += (float)(logsum - logit_target);
            }
            return total / batch;
        }
    }

    // --- More activations ---
    public static class Sigmoid extends Module {
        @Override
        public Mat forward(Mat x) {
            Mat out = new Mat();
            out.rows = x.rows; out.cols = x.cols; out.es = new float[out.rows * out.cols];
            int n = out.es.length;
            for (int i = 0; i < n; i++) out.es[i] = (float)(1.0 / (1.0 + Math.exp(-x.es[i])));
            return out;
        }
    }

    public static class Tanh extends Module {
        @Override
        public Mat forward(Mat x) {
            Mat out = new Mat(); out.rows = x.rows; out.cols = x.cols; out.es = new float[out.rows * out.cols];
            int n = out.es.length;
            for (int i = 0; i < n; i++) out.es[i] = (float)Math.tanh(x.es[i]);
            return out;
        }
    }

    public static class LeakyReLU extends Module {
        private final float negativeSlope;
        public LeakyReLU(float negativeSlope) { this.negativeSlope = negativeSlope; }
        @Override
        public Mat forward(Mat x) {
            Mat out = new Mat(); out.rows = x.rows; out.cols = x.cols; out.es = new float[out.rows * out.cols];
            int n = out.es.length;
            for (int i = 0; i < n; i++) {
                float v = x.es[i]; out.es[i] = v > 0 ? v : negativeSlope * v;
            }
            return out;
        }
    }

    public static class Softplus extends Module {
        @Override
        public Mat forward(Mat x) {
            Mat out = new Mat(); out.rows = x.rows; out.cols = x.cols; out.es = new float[out.rows * out.cols];
            int n = out.es.length;
            for (int i = 0; i < n; i++) out.es[i] = (float)Math.log(1.0 + Math.exp(x.es[i]));
            return out;
        }
    }

    // --- Dropout (stateless mask generation using seed) ---
    public static class Dropout extends Module {
        private final float p;
        private final long seed;
        public Dropout(float p, long seed) { this.p = p; this.seed = seed; }
        @Override
        public Mat forward(Mat x) {
            if (p <= 0f) return x;
            Mat out = new Mat(); out.rows = x.rows; out.cols = x.cols; out.es = new float[out.rows * out.cols];
            java.util.Random r = new java.util.Random(seed);
            float scale = 1.0f / (1.0f - p);
            int n = out.es.length;
            for (int i = 0; i < n; i++) {
                boolean keep = r.nextFloat() >= p;
                out.es[i] = keep ? x.es[i] * scale : 0f;
            }
            return out;
        }
    }

    // --- BatchNorm1d ---
    public static class BatchNorm1d extends Module {
        public int numFeatures;
        public Parameter weight; // gamma
        public Parameter bias;   // beta
        public float[] runningMean;
        public float[] runningVar;
        public float eps = 1e-5f;
        public float momentum = 0.1f;

        public BatchNorm1d(nn outer, int numFeatures, boolean affine) {
            this.numFeatures = numFeatures;
            runningMean = new float[numFeatures];
            runningVar = new float[numFeatures];
            for (int i = 0; i < numFeatures; i++) { runningMean[i] = 0f; runningVar[i] = 1f; }
            if (affine) {
                Mat gw = outer.mat_alloc(1, numFeatures);
                outer.mat_fill(gw, 1.0f);
                weight = new Parameter(gw);
                addParameter("weight", weight);
                Mat gb = outer.mat_alloc(1, numFeatures);
                outer.mat_fill(gb, 0.0f);
                bias = new Parameter(gb);
                addParameter("bias", bias);
            }
        }

        @Override
        public Mat forward(Mat x) {
            if (x.cols != numFeatures) throw new IllegalArgumentException("BatchNorm1d: feature mismatch");
            int batch = x.rows;
            Mat out = new Mat(); out.rows = batch; out.cols = numFeatures; out.es = new float[batch * numFeatures];
            float[] mean = new float[numFeatures];
            float[] var = new float[numFeatures];
            // compute mean
            for (int j = 0; j < numFeatures; j++) {
                float s = 0f;
                for (int i = 0; i < batch; i++) s += x.es[i * numFeatures + j];
                mean[j] = s / batch;
            }
            // compute var
            for (int j = 0; j < numFeatures; j++) {
                float s = 0f;
                for (int i = 0; i < batch; i++) {
                    float d = x.es[i * numFeatures + j] - mean[j]; s += d * d;
                }
                var[j] = s / batch;
            }
            // update running
            for (int j = 0; j < numFeatures; j++) {
                runningMean[j] = momentum * mean[j] + (1 - momentum) * runningMean[j];
                runningVar[j] = momentum * var[j] + (1 - momentum) * runningVar[j];
            }
            // normalize and affine
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    float val = (x.es[i * numFeatures + j] - mean[j]) / (float)Math.sqrt(var[j] + eps);
                    if (weight != null) val = val * weight.data.es[j] + bias.data.es[j];
                    out.es[i * numFeatures + j] = val;
                }
            }
            return out;
        }
    }

        // --- Conv2d (naive im2col per-sample) ---
        public static class Conv2d extends Module {
            public int inChannels, outChannels, kernelH, kernelW;
            public int inH, inW;
            public int strideH=1, strideW=1;
            public int padH=0, padW=0;
            public Parameter weight; // shape: (inC*kh*kw) x outC
            public Parameter bias;   // 1 x (outC)

            public Conv2d(nn outer, int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int stride, int padding, boolean biasFlag) {
                this(inChannels, outChannels, kernelH, kernelW, inH, inW, stride, stride, padding, padding, outer, biasFlag);
            }

            public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int strideH, int strideW, int padH, int padW, nn outer, boolean biasFlag) {
                this.inChannels = inChannels; this.outChannels = outChannels; this.kernelH = kernelH; this.kernelW = kernelW;
                this.inH = inH; this.inW = inW; this.strideH = strideH; this.strideW = strideW; this.padH = padH; this.padW = padW;
                int ksz = inChannels * kernelH * kernelW;
                Mat w = outer.mat_alloc(ksz, outChannels);
                outer.mat_rand(w, -0.08f, 0.08f);
                this.weight = new Parameter(w);
                addParameter("weight", this.weight);
                if (biasFlag) {
                    Mat b = outer.mat_alloc(1, outChannels);
                    outer.mat_fill(b, 0f);
                    this.bias = new Parameter(b);
                    addParameter("bias", this.bias);
                }
            }

            @Override
            public Mat forward(Mat x) {
                // x: rows=batch, cols=inChannels*inH*inW
                int batch = x.rows;
                int outH = (inH + 2*padH - kernelH) / strideH + 1;
                int outW = (inW + 2*padW - kernelW) / strideW + 1;
                Mat out = new Mat(); out.rows = batch; out.cols = outChannels * outH * outW; out.es = new float[out.rows * out.cols];
                int ksz = inChannels * kernelH * kernelW;
                for (int b = 0; b < batch; b++) {
                    // im2col per output location
                    float[] col = new float[outH * outW * ksz];
                    int colIdx = 0;
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            for (int ic = 0; ic < inChannels; ic++) {
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        float val = 0f;
                                        if (ih >= 0 && ih < inH && iw >=0 && iw < inW) {
                                            int idx = b * x.cols + (ic * inH * inW + ih * inW + iw);
                                            val = x.es[idx];
                                        }
                                        col[colIdx++] = val;
                                    }
                                }
                            }
                        }
                    }
                    // multiply col (outH*outW x ksz) with weight (ksz x outC) to get (outH*outW x outC)
                    for (int pos = 0; pos < outH*outW; pos++) {
                        for (int oc = 0; oc < outChannels; oc++) {
                            float sum = 0f;
                            int base = pos * ksz;
                            for (int k = 0; k < ksz; k++) {
                                sum += col[base + k] * weight.data.es[k * weight.data.cols + oc];
                            }
                            int outPos = b * out.cols + (oc * outH * outW + pos);
                            out.es[outPos] = sum + (bias != null ? bias.data.es[oc] : 0f);
                        }
                    }
                }
                return out;
            }

            // convenience constructor: symmetric stride and padding
            public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int stride, int padding, nn outer, boolean biasFlag) {
                this.inChannels = inChannels; this.outChannels = outChannels; this.kernelH = kernelH; this.kernelW = kernelW;
                this.inH = inH; this.inW = inW; this.strideH = stride; this.strideW = stride; this.padH = padding; this.padW = padding;
                int ksz = inChannels * kernelH * kernelW;
                Mat w = outer.mat_alloc(ksz, outChannels);
                outer.mat_rand(w, -0.08f, 0.08f);
                this.weight = new Parameter(w);
                addParameter("weight", this.weight);
                if (biasFlag) {
                    Mat b = outer.mat_alloc(1, outChannels);
                    outer.mat_fill(b, 0f);
                    this.bias = new Parameter(b);
                    addParameter("bias", this.bias);
                }
            }
        }

        // --- MaxPool2d and AvgPool2d (naive) ---
        public static class MaxPool2d extends Module {
            public int kernelH, kernelW, strideH, strideW, padH, padW;
            public int inC, inH, inW;

            public MaxPool2d(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inC, int inH, int inW) {
                this.kernelH = kernelH; this.kernelW = kernelW; this.strideH = strideH; this.strideW = strideW; this.padH = padH; this.padW = padW;
                this.inC = inC; this.inH = inH; this.inW = inW;
            }

            @Override
            public Mat forward(Mat x) {
                int batch = x.rows;
                int outH = (inH + 2*padH - kernelH) / strideH + 1;
                int outW = (inW + 2*padW - kernelW) / strideW + 1;
                Mat out = new Mat(); out.rows = batch; out.cols = inC * outH * outW; out.es = new float[out.rows * out.cols];
                for (int b = 0; b < batch; b++) {
                    for (int c = 0; c < inC; c++) {
                        for (int oh = 0; oh < outH; oh++) {
                            for (int ow = 0; ow < outW; ow++) {
                                float maxv = Float.NEGATIVE_INFINITY;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                            int idx = b * x.cols + (c * inH * inW + ih * inW + iw);
                                            float v = x.es[idx];
                                            if (v > maxv) maxv = v;
                                        }
                                    }
                                }
                                int outIdx = b * out.cols + (c * outH * outW + oh * outW + ow);
                                out.es[outIdx] = maxv;
                            }
                        }
                    }
                }
                return out;
            }
        }

        public static class AvgPool2d extends Module {
            public int kernelH, kernelW, strideH, strideW, padH, padW;
            public int inC, inH, inW;

            public AvgPool2d(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inC, int inH, int inW) {
                this.kernelH = kernelH; this.kernelW = kernelW; this.strideH = strideH; this.strideW = strideW; this.padH = padH; this.padW = padW;
                this.inC = inC; this.inH = inH; this.inW = inW;
            }

            @Override
            public Mat forward(Mat x) {
                int batch = x.rows;
                int outH = (inH + 2*padH - kernelH) / strideH + 1;
                int outW = (inW + 2*padW - kernelW) / strideW + 1;
                Mat out = new Mat(); out.rows = batch; out.cols = inC * outH * outW; out.es = new float[out.rows * out.cols];
                for (int b = 0; b < batch; b++) {
                    for (int c = 0; c < inC; c++) {
                        for (int oh = 0; oh < outH; oh++) {
                            for (int ow = 0; ow < outW; ow++) {
                                float sumv = 0f; int cnt=0;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                            int idx = b * x.cols + (c * inH * inW + ih * inW + iw);
                                            sumv += x.es[idx]; cnt++;
                                        }
                                    }
                                }
                                int outIdx = b * out.cols + (c * outH * outW + oh * outW + ow);
                                out.es[outIdx] = cnt>0 ? sumv / cnt : 0f;
                            }
                        }
                    }
                }
                return out;
            }
        }

        // --- Zero padding utility ---
        public static class ZeroPad2d extends Module {
            public int padH, padW, inC, inH, inW;
            public ZeroPad2d(int padH, int padW, int inC, int inH, int inW) { this.padH = padH; this.padW = padW; this.inC = inC; this.inH = inH; this.inW = inW; }
            @Override
            public Mat forward(Mat x) {
                int outH = inH + 2*padH; int outW = inW + 2*padW;
                Mat out = new Mat(); out.rows = x.rows; out.cols = inC * outH * outW; out.es = new float[out.rows * out.cols];
                for (int b = 0; b < x.rows; b++) {
                    for (int c = 0; c < inC; c++) {
                        for (int h = 0; h < inH; h++) {
                            for (int w = 0; w < inW; w++) {
                                float v = x.es[b * x.cols + (c * inH * inW + h * inW + w)];
                                int outIdx = b * out.cols + (c * outH * outW + (h + padH) * outW + (w + padW));
                                out.es[outIdx] = v;
                            }
                        }
                    }
                }
                return out;
            }
        }

}