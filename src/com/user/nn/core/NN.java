package com.user.nn.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public class NN {
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
        if (a.cols != b.rows)
            throw new IllegalArgumentException("Incompatible matrix dimensions for dot: a.cols must equal b.rows");
        if (dst.rows != a.rows || dst.cols != b.cols)
            throw new IllegalArgumentException("Destination matrix has wrong dimensions");
        Tensor ta = Torch.fromMat(a);
        Tensor tb = Torch.fromMat(b);
        Tensor tc = Torch.matmul(ta, tb);
        // copy back
        if (tc.shape.length != 2 || tc.shape[0] != dst.rows || tc.shape[1] != dst.cols)
            throw new IllegalStateException("matmul result shape mismatch");
        System.arraycopy(tc.data, 0, dst.es, 0, dst.es.length);
    }

    public void mat_sum(Mat dst, Mat a) {
        if (dst.rows != a.rows || dst.cols != a.cols)
            throw new IllegalArgumentException("Matrices must have the same dimensions for addition");
        Tensor ta = Torch.fromMat(dst);
        Tensor tb = Torch.fromMat(a);
        Tensor tr = Torch.add(ta, tb);
        System.arraycopy(tr.data, 0, dst.es, 0, dst.es.length);
    }

    public void mat_sub(Mat dst, Mat a) {
        if (dst.rows != a.rows || dst.cols != a.cols)
            throw new IllegalArgumentException("Matrices must have the same dimensions for subtraction");
        Tensor ta = Torch.fromMat(dst);
        Tensor tb = Torch.fromMat(a);
        Tensor tr = Torch.sub(ta, tb);
        System.arraycopy(tr.data, 0, dst.es, 0, dst.es.length);
    }

    public void mat_print(Mat m) {
        Tensor t = Torch.fromMat(m);
        System.out.println(t.toString());
    }

    public void mat_rand(Mat m, float min, float max) {
        Tensor t = Torch.rand(new int[] { m.rows, m.cols });
        // scale to [min,max)
        float range = max - min;
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = min + t.data[i] * range;
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    public void mat_fill(Mat m, float value) {
        Tensor t = Torch.full(new int[] { m.rows, m.cols }, value);
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // Elementwise apply (in-place) using a lambda-like helper interface
    public interface ElemOp {
        float apply(float x);
    }

    public void mat_apply_inplace(Mat m, ElemOp op) {
        // apply elementwise via Tensor
        Tensor t = Torch.fromMat(m);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = op.apply(t.data[i]);
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // Deterministic random fill
    public void mat_rand_seed(Mat m, long seed, float min, float max) {
        // deterministic rand via Torch.randn seeded by seed
        Torch.manual_seed(seed);
        Tensor t = Torch.rand(new int[] { m.rows, m.cols });
        float range = max - min;
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = min + t.data[i] * range;
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // CSV read/write utilities (rows lines, comma-separated)
    public void writeMatCSV(Mat m, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                sb.append(m.es[i * m.cols + j]);
                if (j + 1 < m.cols)
                    sb.append(',');
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
        if (lines.size() == 0)
            return null;
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
        public Mat data; // legacy
        public Tensor tensorData; // modern
        public boolean requiresGrad = true;

        public Parameter(Mat data) {
            this.data = data;
            this.tensorData = Torch.fromMat(data);
            this.tensorData.requires_grad = this.requiresGrad;
        }

        public Parameter(Tensor tensor) {
            this.tensorData = tensor;
            this.tensorData.requires_grad = this.requiresGrad;
        }

        public Tensor getTensor() {
            this.tensorData.requires_grad = this.requiresGrad;
            return this.tensorData;
        }

        public Tensor getGrad() {
            return this.tensorData.grad;
        }

        public Parameter toGPU() {
            if (this.tensorData != null)
                this.tensorData.toGPU();
            return this;
        }

        public Parameter toCPU() {
            if (this.tensorData != null)
                this.tensorData.toCPU();
            return this;
        }
    }

    public static abstract class Module {
        protected Map<String, Module> children = new LinkedHashMap<>();
        protected Map<String, Parameter> params = new LinkedHashMap<>();
        protected boolean training = true;

        public void train() {
            this.training = true;
            for (Module m : children.values()) {
                m.train();
            }
        }

        public void eval() {
            this.training = false;
            for (Module m : children.values()) {
                m.eval();
            }
        }

        public void toGPU() {
            for (Parameter p : params.values()) {
                p.toGPU();
            }
            for (Module m : children.values()) {
                m.toGPU();
            }
        }

        public void toCPU() {
            for (Parameter p : params.values()) {
                p.toCPU();
            }
            for (Module m : children.values()) {
                m.toCPU();
            }
        }

        public boolean is_training() {
            return training;
        }

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

        public long countParameters() {
            long total = 0;
            for (Parameter p : parameters()) {
                total += p.getTensor().numel();
            }
            return total;
        }

        public List<Module> modules() {
            List<Module> out = new ArrayList<>();
            out.addAll(children.values());
            return out;
        }

        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException(
                    this.getClass().getSimpleName() + " does not implement forward(Tensor) directly");
        }

        public Mat forward(Mat x) {
            Tensor t = Torch.fromMat(x);
            Tensor out = forward(t);
            NN outer = new NN();
            Mat m;
            if (out.dim() == 2)
                m = outer.mat_alloc(out.shape[0], out.shape[1]);
            else
                m = outer.mat_alloc(out.shape[0], out.numel() / out.shape[0]);
            System.arraycopy(out.data, 0, m.es, 0, m.es.length);
            return m;
        }

        public Tensor apply(Tensor x) {
            return forward(x);
        }

        public Mat apply(Mat x) {
            return forward(x);
        }
    }

    public static class Sequential extends Module {
        private final List<Module> list = new ArrayList<>();

        public Sequential() {
        }

        public void add(Module m) {
            String name = "" + list.size();
            list.add(m);
            addModule(name, m);
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor out = x;
            for (int i = 0; i < list.size(); i++) {
                Module m = list.get(i);
                
                // Kernel Fusion: detect Conv2d(bias=true) + ReLU on GPU
                if (out.isGPU() && m instanceof Conv2d && i + 1 < list.size() && list.get(i + 1) instanceof ReLU) {
                    Conv2d conv = (Conv2d) m;
                    if (conv.bias != null) {
                        // Fused Conv2d + Bias + ReLU forward
                        int batch = out.shape[0];
                        int outH = (conv.inH + 2 * conv.padH - conv.kernelH) / conv.strideH + 1;
                        int outW = (conv.inW + 2 * conv.padW - conv.kernelW) / conv.strideW + 1;
                        int ksz = conv.inChannels * conv.kernelH * conv.kernelW;
                        int outSize = conv.outChannels * outH * outW;

                        Tensor wt = conv.weight.getTensor();
                        Tensor bt = conv.bias.getTensor();
                        Tensor fusedOut = new Tensor(batch, outSize);
                        fusedOut.toGPU();

                        Tensor wtT = new Tensor(new int[]{conv.outChannels, ksz});
                        wtT.toGPU();
                        CUDAOps.transpose(wt, wtT);
                        
                        CUDAOps.conv2dBiasReluForward(out, wtT, bt, fusedOut,
                            conv.inChannels, conv.inH, conv.inW,
                            conv.kernelH, conv.kernelW,
                            conv.outChannels, outH, outW,
                            conv.padH, conv.padW, conv.strideH, conv.strideW);
                        
                        wtT.close();
                        
                        // Attach autograd (same as Conv2d backward, ReLU backward will be handled by the combined output)
                        if (Torch.is_grad_enabled() && (out.requires_grad || wt.requires_grad || bt.requires_grad)) {
                            fusedOut.requires_grad = true;
                            final Tensor convInput = out;
                            fusedOut.grad_fn = new Tensor.GradFn(convInput, wt, bt) {
                                public void apply(Tensor outGrad) {
                                    // ReLU backward: mask gradient where fused output <= 0
                                    fusedOut.toCPU();
                                    outGrad.toCPU();
                                    for (int j = 0; j < fusedOut.data.length; j++) {
                                        if (fusedOut.data[j] <= 0) outGrad.data[j] = 0;
                                    }
                                    if (convInput.isGPU()) outGrad.toGPU();
                                    
                                    // Conv2d backward (delegate to original backward logic)
                                    conv.forward(convInput).grad_fn.apply(outGrad);
                                }
                            };
                        }
                        
                        out = fusedOut;
                        i++; // Skip the next ReLU layer
                        continue;
                    }
                }
                
                out = m.forward(out);
            }
            return out;
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

        public Module get(int idx) {
            return list.get(idx);
        }

        @Override
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("ModuleList does not implement forward directly");
        }

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
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("ModuleDict does not implement forward directly");
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
        public Parameter bias; // shape: 1 x outFeatures (row vector)

        public Linear(NN outer, int inFeatures, int outFeatures, boolean useBias) {
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
        public Tensor forward(Tensor input) {
            if (input.shape[input.shape.length - 1] != inFeatures) {
                throw new IllegalArgumentException("Input features mismatch: expected " + inFeatures + " got "
                        + input.shape[input.shape.length - 1]);
            }
            Tensor w = this.weight.getTensor();
            Tensor out = Torch.matmul(input, w);
            if (this.bias != null) {
                Tensor b = this.bias.getTensor();
                out = Torch.add(out, b);
            }
            return out;
        }
    }

    public static class Embedding extends Module {
        public int numEmbeddings;
        public int embeddingDim;
        public Parameter weight;

        public Embedding(NN outer, int numEmbeddings, int embeddingDim) {
            this.numEmbeddings = numEmbeddings;
            this.embeddingDim = embeddingDim;
            Mat w = outer.mat_alloc(numEmbeddings, embeddingDim);
            float k = (float) Math.sqrt(1.0 / embeddingDim);
            outer.mat_rand(w, -k, k);
            this.weight = new Parameter(w);
            addParameter("weight", this.weight);
        }

        @Override
        public Tensor forward(Tensor indices) {
            indices.toCPU();
            Tensor w = weight.getTensor();
            w.toCPU();
            int[] idxShape = indices.shape;
            int numIdx = indices.numel();
            int[] outShape = new int[idxShape.length + 1];
            System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
            outShape[idxShape.length] = embeddingDim;

            Tensor out = new Tensor(outShape);
            if (w.isGPU() || indices.isGPU()) out.toGPU();

            for (int i = 0; i < numIdx; i++) {
                int idx = (int) indices.data[i];
                if (idx < 0 || idx >= numEmbeddings) {
                    throw new IndexOutOfBoundsException("Embedding index out of range: " + idx);
                }
                System.arraycopy(w.data, idx * embeddingDim, out.data, i * embeddingDim, embeddingDim);
            }

            if (Torch.is_grad_enabled() && w.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(weight.getTensor()) {
                    public void apply(Tensor outGrad) {
                        Tensor gw = new Tensor(w.shape);
                        for (int i = 0; i < numIdx; i++) {
                            int idx = (int) indices.data[i];
                            for (int d = 0; d < embeddingDim; d++) {
                                gw.data[idx * embeddingDim + d] += outGrad.data[i * embeddingDim + d];
                            }
                        }
                        w.backwardStep(gw);
                    }
                };
            }
            return out;
        }
    }

    public static class ReLU extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return Torch.relu(x);
        }
    }

    // Functional utilities
    public static class F {
        public static Mat relu(NN outer, Mat x) {
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
            if (logits.rows != targets.length)
                throw new IllegalArgumentException("cross_entropy: batch size mismatch");
            int batch = logits.rows;
            int classes = logits.cols;
            float total = 0f;
            for (int i = 0; i < batch; i++) {
                // find max for numerical stability
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < classes; j++)
                    if (logits.es[i * classes + j] > max)
                        max = logits.es[i * classes + j];
                double sum = 0.0;
                for (int j = 0; j < classes; j++)
                    sum += Math.exp(logits.es[i * classes + j] - max);
                double logsum = Math.log(sum) + max;
                int t = targets[i];
                double logit_target = logits.es[i * classes + t];
                total += (float) (logsum - logit_target);
            }
            return total / batch;
        }

        // autograd-aware cross entropy returning scalar Tensor
        public static Tensor cross_entropy_tensor(Tensor logits, int[] targets) {
            logits.toCPU();
            int batch = logits.shape[0];
            int classes = logits.shape[1];
            // compute softmax per-row and loss
            Tensor out = new Tensor(1);
            float total = 0f;
            float[][] soft = new float[batch][classes];
            for (int i = 0; i < batch; i++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < classes; j++)
                    if (logits.data[i * classes + j] > max)
                        max = logits.data[i * classes + j];
                double sum = 0.0;
                for (int j = 0; j < classes; j++) {
                    double e = Math.exp(logits.data[i * classes + j] - max);
                    soft[i][j] = (float) e;
                    sum += e;
                }
                for (int j = 0; j < classes; j++)
                    soft[i][j] /= sum;
                int t = targets[i];
                total += (float) (-Math.log(Math.max(1e-12, soft[i][t])));
            }
            out.data[0] = total / batch;
            if (logits.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(logits) {
                    public void apply(Tensor outGrad) {
                        float scale = outGrad.data[0] / batch;
                        Tensor g = new Tensor(logits.shape);
                        for (int i = 0; i < batch; i++) {
                            for (int j = 0; j < classes; j++) {
                                float one = (j == targets[i]) ? 1f : 0f;
                                g.data[i * classes + j] = (soft[i][j] - one) * scale;
                            }
                        }
                        logits.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /**
         * Negative log-likelihood loss. Input should be log-probabilities [batch,
         * classes].
         */
        public static Tensor nll_loss(Tensor logProbs, int[] targets) {
            logProbs.toCPU();
            int batch = logProbs.shape[0];
            int classes = logProbs.shape[1];
            Tensor out = new Tensor(1);
            float total = 0f;
            for (int i = 0; i < batch; i++) {
                total += -logProbs.data[i * classes + targets[i]];
            }
            out.data[0] = total / batch;
            if (logProbs.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(logProbs) {
                    public void apply(Tensor outGrad) {
                        float scale = outGrad.data[0] / batch;
                        Tensor g = new Tensor(logProbs.shape);
                        for (int i = 0; i < batch; i++) {
                            g.data[i * classes + targets[i]] = -scale;
                        }
                        logProbs.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /** Autograd-aware MSE loss returning scalar Tensor. */
        public static Tensor mse_loss_tensor(Tensor pred, Tensor target) {
            pred.toCPU();
            target.toCPU();
            int n = pred.numel();
            Tensor out = new Tensor(1);
            float sum = 0f;
            for (int i = 0; i < n; i++) {
                float d = pred.data[i] - target.data[i];
                sum += d * d;
            }
            out.data[0] = sum / n;
            if (pred.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(pred) {
                    public void apply(Tensor outGrad) {
                        Tensor g = new Tensor(pred.shape);
                        float scale = 2f * outGrad.data[0] / n;
                        for (int i = 0; i < n; i++) {
                            g.data[i] = (pred.data[i] - target.data[i]) * scale;
                        }
                        pred.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /** Huber loss (smooth L1). Quadratic for |error|<delta, linear otherwise. */
        public static Tensor huber_loss(Tensor pred, Tensor target, float delta) {
            int n = pred.data.length;
            Tensor out = new Tensor(1);
            float sum = 0f;
            for (int i = 0; i < n; i++) {
                float d = pred.data[i] - target.data[i];
                if (Math.abs(d) <= delta) {
                    sum += 0.5f * d * d;
                } else {
                    sum += delta * (Math.abs(d) - 0.5f * delta);
                }
            }
            out.data[0] = sum / n;
            if (pred.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(pred) {
                    public void apply(Tensor outGrad) {
                        Tensor g = new Tensor(pred.shape);
                        float scale = outGrad.data[0] / n;
                        for (int i = 0; i < n; i++) {
                            float d = pred.data[i] - target.data[i];
                            if (Math.abs(d) <= delta) {
                                g.data[i] = d * scale;
                            } else {
                                g.data[i] = delta * Math.signum(d) * scale;
                            }
                        }
                        pred.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /** L1 Loss (Mean Absolute Error). */
        public static Tensor l1_loss(Tensor pred, Tensor target) {
            int n = pred.data.length;
            Tensor out = new Tensor(1);
            float sum = 0f;
            for (int i = 0; i < n; i++) {
                sum += Math.abs(pred.data[i] - target.data[i]);
            }
            out.data[0] = sum / n;
            if (pred.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(pred) {
                    public void apply(Tensor outGrad) {
                        Tensor g = new Tensor(pred.shape);
                        float scale = outGrad.data[0] / n;
                        for (int i = 0; i < n; i++) {
                            g.data[i] = Math.signum(pred.data[i] - target.data[i]) * scale;
                        }
                        pred.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /** Binary Cross Entropy Loss. Input (probs) and target should be same shape. */
        public static Tensor binary_cross_entropy(Tensor input, Tensor target) {
            int n = input.data.length;
            Tensor out = new Tensor(1);
            float total = 0f;
            for (int i = 0; i < n; i++) {
                float h = input.data[i];
                float y = target.data[i];
                h = Math.max(1e-12f, Math.min(1f - 1e-12f, h));
                total += -(y * (float) Math.log(h) + (1f - y) * (float) Math.log(1f - h));
            }
            out.data[0] = total / n;
            if (input.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(input) {
                    public void apply(Tensor outGrad) {
                        Tensor g = new Tensor(input.shape);
                        float scale = outGrad.data[0] / n;
                        for (int i = 0; i < n; i++) {
                            float h = Math.max(1e-12f, Math.min(1f - 1e-12f, input.data[i]));
                            float y = target.data[i];
                            g.data[i] = ((h - y) / (h * (1f - h) + 1e-12f)) * scale;
                        }
                        input.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /** BCE With Logits Loss (combined sigmoid + BCELoss for stability). */
        public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target) {
            int n = input.data.length;
            Tensor out = new Tensor(1);
            float total = 0f;
            for (int i = 0; i < n; i++) {
                float x = input.data[i];
                float y = target.data[i];
                if (x > 0) {
                    total += x * (1 - y) + (float) Math.log(1 + Math.exp(-x));
                } else {
                    total += -x * y + (float) Math.log(1 + Math.exp(x));
                }
            }
            out.data[0] = total / n;
            if (input.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(input) {
                    public void apply(Tensor outGrad) {
                        Tensor g = new Tensor(input.shape);
                        float scale = outGrad.data[0] / n;
                        for (int i = 0; i < n; i++) {
                            float sig = (float) (1.0 / (1.0 + Math.exp(-input.data[i])));
                            g.data[i] = (sig - target.data[i]) * scale;
                        }
                        input.backwardStep(g);
                    }
                };
            }
            return out;
        }

        /** KL Divergence Loss. input is log-probs, target is probs. */
        public static Tensor kl_div(Tensor input, Tensor target) {
            int n = input.data.length;
            Tensor out = new Tensor(1);
            float total = 0f;
            for (int i = 0; i < n; i++) {
                float logP = input.data[i];
                float Q = target.data[i];
                if (Q > 0) {
                    total += Q * ((float) Math.log(Q) - logP);
                }
            }
            out.data[0] = total / n;
            if (input.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(input) {
                    public void apply(Tensor outGrad) {
                        Tensor g = new Tensor(input.shape);
                        float scale = outGrad.data[0] / n;
                        for (int i = 0; i < n; i++) {
                            g.data[i] = -target.data[i] * scale;
                        }
                        input.backwardStep(g);
                    }
                };
            }
            return out;
        }

        public static Tensor cosine_similarity(Tensor x1, Tensor x2, int dim, float eps) {
            return Torch.cosine_similarity(x1, x2, dim, eps);
        }

        public static Tensor pairwise_distance(Tensor x1, Tensor x2, float p, float eps) {
            return Torch.pairwise_distance(x1, x2, p, eps);
        }

        public static Tensor softmax(Tensor x, int dim) {
            return Torch.softmax(x, dim);
        }

        public static Tensor log_softmax(Tensor x, int dim) {
            return Torch.log_softmax(x, dim);
        }

        public static Tensor gelu(Tensor x) {
            return Torch.gelu(x);
        }

        public static Tensor elu(Tensor x, float alpha) {
            return Torch.elu(x, alpha);
        }

        public static Tensor silu(Tensor x) {
            return Torch.silu(x);
        }

        public static Tensor max_pool1d(Tensor x, int kernel, int stride, int pad) {
            return Torch.max_pool1d(x, kernel, stride, pad);
        }

        public static Tensor avg_pool1d(Tensor x, int kernel, int stride, int pad) {
            return Torch.avg_pool1d(x, kernel, stride, pad);
        }

        public static Tensor adaptive_avg_pool2d(Tensor x, int outputH, int outputW) {
            return Torch.adaptive_avg_pool2d(x, new int[] { outputH, outputW });
        }

        public static Tensor pad(Tensor x, int[] pad, String mode, float value) {
            return Torch.pad(x, pad, mode, value);
        }

        public static Tensor conv1d(Tensor x, Tensor weight, Tensor bias, int stride, int padding) {
            return Torch.conv1d(x, weight, bias, stride, padding);
        }

        public static Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, Tensor bias) {
            return Torch.bilinear(x1, x2, weight, bias);
        }

        public static Tensor one_hot(Tensor indices, int numClasses) {
            return Torch.one_hot(indices, numClasses);
        }

        public static Tensor embedding(Tensor weight, Tensor indices) {
            return Torch.embedding(weight, indices);
        }

        public static Tensor dropout(Tensor x, float p, boolean training) {
            return Torch.dropout(x, p, training);
        }
    }

    // --- More activations ---
    public static class Sigmoid extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return Torch.sigmoid(x);
        }
    }

    public static class Tanh extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return Torch.tanh(x);
        }
    }

    public static class LeakyReLU extends Module {
        private final float negativeSlope;

        public LeakyReLU(float negativeSlope) {
            this.negativeSlope = negativeSlope;
        }

        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            Tensor out = new Tensor(x.shape);
            for (int i = 0; i < x.data.length; i++) {
                float v = x.data[i];
                out.data[i] = v > 0 ? v : negativeSlope * v;
            }
            if (x.isGPU()) out.toGPU();
            if (Torch.is_grad_enabled() && x.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x) {
                    public void apply(Tensor outGrad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int i = 0; i < gx.data.length; i++) {
                            gx.data[i] = (x.data[i] > 0 ? 1f : negativeSlope) * outGrad.data[i];
                        }
                        x.backwardStep(gx);
                    }
                };
            }
            return out;
        }
    }

    public static class GELU extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return Torch.gelu(x);
        }
    }

    public static class ELU extends Module {
        public float alpha;

        public ELU(float alpha) {
            this.alpha = alpha;
        }

        public ELU() {
            this(1.0f);
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.elu(x, alpha);
        }
    }

    public static class SiLU extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return Torch.silu(x);
        }
    }

    public static class Softmax extends Module {
        public int dim;

        public Softmax(int dim) {
            this.dim = dim;
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.softmax(x, dim);
        }
    }

    public static class LogSoftmax extends Module {
        public int dim;

        public LogSoftmax(int dim) {
            this.dim = dim;
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.log_softmax(x, dim);
        }
    }

    public static class Softplus extends Module {
        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            Tensor out = new Tensor(x.shape);
            for (int i = 0; i < x.data.length; i++) {
                out.data[i] = (float) Math.log(1.0 + Math.exp(x.data[i]));
            }
            if (x.isGPU()) out.toGPU();
            if (Torch.is_grad_enabled() && x.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x) {
                    public void apply(Tensor outGrad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int i = 0; i < gx.data.length; i++) {
                            float expX = (float) Math.exp(x.data[i]);
                            gx.data[i] = (expX / (1.0f + expX)) * outGrad.data[i];
                        }
                        x.backwardStep(gx);
                    }
                };
            }
            return out;
        }
    }

    // --- Loss Modules ---
    public static class L1Loss extends Module {
        public Tensor forward(Tensor x, Tensor target) {
            return F.l1_loss(x, target);
        }
    }

    public static class BCELoss extends Module {
        public Tensor forward(Tensor x, Tensor target) {
            return F.binary_cross_entropy(x, target);
        }
    }

    public static class BCEWithLogitsLoss extends Module {
        public Tensor forward(Tensor x, Tensor target) {
            return F.binary_cross_entropy_with_logits(x, target);
        }
    }

    public static class KLDivLoss extends Module {
        public Tensor forward(Tensor x, Tensor target) {
            return F.kl_div(x, target);
        }
    }

    public static class CrossEntropyLoss extends Module {
        public Tensor forward(Tensor x, int[] targets) {
            return F.cross_entropy_tensor(x, targets);
        }
    }

    // --- Distance Modules ---
    public static class CosineSimilarity extends Module {
        public int dim;
        public float eps;

        public CosineSimilarity(int dim, float eps) {
            this.dim = dim;
            this.eps = eps;
        }

        public CosineSimilarity() {
            this(1, 1e-8f);
        }

        public Tensor forward(Tensor x1, Tensor x2) {
            return F.cosine_similarity(x1, x2, dim, eps);
        }
    }

    public static class PairwiseDistance extends Module {
        public float p;
        public float eps;

        public PairwiseDistance(float p, float eps) {
            this.p = p;
            this.eps = eps;
        }

        public PairwiseDistance() {
            this(2.0f, 1e-6f);
        }

        public Tensor forward(Tensor x1, Tensor x2) {
            return F.pairwise_distance(x1, x2, p, eps);
        }
    }

    // --- Dropout ---
    public static class Dropout extends Module {
        public float p;

        public Dropout(float p) {
            this.p = p;
        }

        public Dropout(float p, long seed) {
            this.p = p;
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.dropout(x, p, training);
        }
    }

    // --- BatchNorm1d ---
    public static class BatchNorm1d extends Module {
        public int numFeatures;
        public Parameter weight; // gamma
        public Parameter bias; // beta
        public float[] runningMean;
        public float[] runningVar;
        public float eps = 1e-5f;
        public float momentum = 0.1f;

        public BatchNorm1d(NN outer, int numFeatures, boolean affine) {
            this.numFeatures = numFeatures;
            runningMean = new float[numFeatures];
            runningVar = new float[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                runningMean[i] = 0f;
                runningVar[i] = 1f;
            }
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
            if (x.cols != numFeatures)
                throw new IllegalArgumentException("BatchNorm1d: feature mismatch");
            int batch = x.rows;
            Mat out = new Mat();
            out.rows = batch;
            out.cols = numFeatures;
            out.es = new float[batch * numFeatures];

            float[] useMean;
            float[] useVar;

            if (training) {
                float[] mean = new float[numFeatures];
                float[] var = new float[numFeatures];
                // compute mean
                for (int j = 0; j < numFeatures; j++) {
                    float s = 0f;
                    for (int i = 0; i < batch; i++)
                        s += x.es[i * numFeatures + j];
                    mean[j] = s / batch;
                }
                // compute var
                for (int j = 0; j < numFeatures; j++) {
                    float s = 0f;
                    for (int i = 0; i < batch; i++) {
                        float d = x.es[i * numFeatures + j] - mean[j];
                        s += d * d;
                    }
                    var[j] = s / batch;
                }
                // update running
                for (int j = 0; j < numFeatures; j++) {
                    runningMean[j] = momentum * mean[j] + (1 - momentum) * runningMean[j];
                    runningVar[j] = (momentum * var[j]) + (1 - momentum) * runningVar[j];
                }
                useMean = mean;
                useVar = var;
            } else {
                useMean = runningMean;
                useVar = runningVar;
            }

            // normalize and affine
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    float val = (x.es[i * numFeatures + j] - useMean[j]) / (float) Math.sqrt(useVar[j] + eps);
                    if (weight != null)
                        val = val * weight.data.es[j] + bias.data.es[j];
                    out.es[i * numFeatures + j] = val;
                }
            }
            return out;
        }
    }

    // --- BatchNorm2d (with CPU Autograd) ---
    public static class BatchNorm2d extends Module {
        public int numFeatures;
        public Parameter weight; // gamma 
        public Parameter bias; // beta 
        public float[] runningMean;
        public float[] runningVar;
        public float eps = 1e-5f;
        public float momentum = 0.1f;

        public BatchNorm2d(NN outer, int numFeatures) {
            this(numFeatures, true, outer);
        }

        public BatchNorm2d(int numFeatures, boolean affine, NN outer) {
            this.numFeatures = numFeatures;
            runningMean = new float[numFeatures];
            runningVar = new float[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                runningMean[i] = 0f;
                runningVar[i] = 1f;
            }
            if (affine && outer != null) {
                Mat gw = outer.mat_alloc(1, numFeatures);
                outer.mat_fill(gw, 1.0f);
                weight = new Parameter(gw);
                addParameter("weight", weight);
                
                Mat gb = outer.mat_alloc(1, numFeatures);
                outer.mat_fill(gb, 0.0f);
                bias = new Parameter(gb);
                addParameter("bias", bias);
            } else if (affine && outer == null) {
                float[] gw = new float[numFeatures];
                float[] gb = new float[numFeatures];
                for(int i=0; i<numFeatures; i++) { gw[i] = 1f; gb[i] = 0f; }
                weight = new Parameter(new Tensor(gw, numFeatures));
                addParameter("weight", weight);
                bias = new Parameter(new Tensor(gb, numFeatures));
                addParameter("bias", bias);
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            if (x.shape.length != 4) throw new IllegalArgumentException("BatchNorm2d requires 4D tensor [batch, C, H, W]");
            if (x.shape[1] != numFeatures) throw new IllegalArgumentException("BatchNorm2d channel mismatch: expected " + numFeatures + " got " + x.shape[1]);
            
            int batch = x.shape[0];
            int c = x.shape[1];
            int hw = x.shape[2] * x.shape[3];
            int chw = c * hw;
            int count = batch * hw;
            
            float[] useMean = new float[c];
            float[] useVar = new float[c];
            float[] invStd = new float[c];
            
            Tensor out = new Tensor(x.shape);
            
            boolean wasGPU = x.isGPU();
            if (wasGPU) x.toCPU(); // CPU fallback calculation

            if (training) {
                for (int ic = 0; ic < c; ic++) {
                    float sum = 0f;
                    for (int b = 0; b < batch; b++) {
                        int bOff = b * chw + ic * hw;
                        for (int i = 0; i < hw; i++) sum += x.data[bOff + i];
                    }
                    useMean[ic] = sum / count;
                }
                for (int ic = 0; ic < c; ic++) {
                    float sum = 0f;
                    float m = useMean[ic];
                    for (int b = 0; b < batch; b++) {
                        int bOff = b * chw + ic * hw;
                        for (int i = 0; i < hw; i++) {
                            float d = x.data[bOff + i] - m;
                            sum += d * d;
                        }
                    }
                    useVar[ic] = sum / count;
                    runningMean[ic] = momentum * useMean[ic] + (1 - momentum) * runningMean[ic];
                    runningVar[ic]  = momentum * useVar[ic] + (1 - momentum) * runningVar[ic];
                }
            } else {
                System.arraycopy(runningMean, 0, useMean, 0, c);
                System.arraycopy(runningVar, 0, useVar, 0, c);
            }

            float[] wData = weight != null ? weight.getTensor().data : null;
            float[] bData = bias != null ? bias.getTensor().data : null;

            for (int ic = 0; ic < c; ic++) {
                invStd[ic] = 1.0f / (float) Math.sqrt(useVar[ic] + eps);
                float m = useMean[ic];
                float istd = invStd[ic];
                float gw = wData != null ? wData[ic] : 1.0f;
                float gb = bData != null ? bData[ic] : 0.0f;
                
                for (int b = 0; b < batch; b++) {
                    int bOff = b * chw + ic * hw;
                    for (int i = 0; i < hw; i++) {
                        float norm = (x.data[bOff + i] - m) * istd;
                        out.data[bOff + i] = norm * gw + gb;
                    }
                }
            }
            
            if (Torch.is_grad_enabled() && (x.requires_grad || (weight != null && weight.getTensor().requires_grad))) {
                out.requires_grad = true;
                final float[] saveMean = useMean, saveInvStd = invStd;
                out.grad_fn = new Tensor.GradFn(x, weight != null ? weight.getTensor() : null, bias != null ? bias.getTensor() : null) {
                    public void apply(Tensor outGrad) {
                        if (outGrad.isGPU()) outGrad.toCPU();
                        Tensor gx = new Tensor(x.shape);
                        Tensor wt = weight != null ? weight.getTensor() : null;
                        Tensor bt = bias != null ? bias.getTensor() : null;
                        
                        float[] dGamma = wt != null && wt.requires_grad ? new float[c] : null;
                        float[] dBeta = bt != null && bt.requires_grad ? new float[c] : null;

                        for (int ic = 0; ic < c; ic++) {
                            float m = saveMean[ic];
                            float istd = saveInvStd[ic];
                            float gamma = wt != null ? wt.data[ic] : 1.0f;

                            float sum_dy = 0f, sum_dy_x_hat = 0f;

                            for (int b = 0; b < batch; b++) {
                                int bOff = b * chw + ic * hw;
                                for (int i = 0; i < hw; i++) {
                                    float dy = outGrad.data[bOff + i];
                                    float x_hat = (x.data[bOff + i] - m) * istd;
                                    sum_dy += dy;
                                    sum_dy_x_hat += dy * x_hat;
                                    
                                    if (dGamma != null) dGamma[ic] += dy * x_hat;
                                    if (dBeta != null) dBeta[ic] += dy;
                                }
                            }
                            
                            if (x.requires_grad) {
                                float c1 = gamma * istd / count;
                                float c2 = (float) count;
                                for (int b = 0; b < batch; b++) {
                                    int bOff = b * chw + ic * hw;
                                    for (int i = 0; i < hw; i++) {
                                        float dx_hat = outGrad.data[bOff + i];
                                        float x_hat = (x.data[bOff + i] - m) * istd;
                                        float dval = c1 * (c2 * dx_hat - sum_dy - x_hat * sum_dy_x_hat);
                                        gx.data[bOff + i] += dval;
                                    }
                                }
                            }
                        }
                        
                        if (x.requires_grad) {
                            if (wasGPU) gx.toGPU();
                            x.backwardStep(gx);
                        }
                        if (wt != null && wt.requires_grad) wt.backwardStep(new Tensor(dGamma, wt.shape));
                        if (bt != null && bt.requires_grad) bt.backwardStep(new Tensor(dBeta, bt.shape));
                    }
                };
            }

            if (wasGPU) {
                x.toGPU(); // restore
                out.toGPU();
            }
            return out;
        }
    }

    // --- Conv2d (naive im2col per-sample) ---
    public static class Conv2d extends Module {
        public int inChannels, outChannels, kernelH, kernelW;
        public int inH, inW;
        public int strideH = 1, strideW = 1;
        public int padH = 0, padW = 0;
        public Parameter weight; // shape: (inC*kh*kw) x outC
        public Parameter bias; // 1 x (outC)

        public Conv2d(NN outer, int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int stride,
                int padding, boolean biasFlag) {
            this(inChannels, outChannels, kernelH, kernelW, inH, inW, stride, stride, padding, padding, outer,
                    biasFlag);
        }

        public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int strideH,
                int strideW, int padH, int padW, NN outer, boolean biasFlag) {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelH = kernelH;
            this.kernelW = kernelW;
            this.inH = inH;
            this.inW = inW;
            this.strideH = strideH;
            this.strideW = strideW;
            this.padH = padH;
            this.padW = padW;
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
        public Tensor forward(Tensor x) {
            int batch = x.shape[0];
            int outH = (inH + 2 * padH - kernelH) / strideH + 1;
            int outW = (inW + 2 * padW - kernelW) / strideW + 1;
            int ksz = inChannels * kernelH * kernelW;
            int inSize = inChannels * inH * inW;
            int outSize = outChannels * outH * outW;

            Tensor wt = this.weight.getTensor();
            Tensor bt = this.bias != null ? this.bias.getTensor() : null;
            Tensor out = new Tensor(batch, outSize);
            float[][] colAll = null;

            if (x.isGPU()) {
                out.toGPU();
                // cuDNN expects weights in [outC, inC*kH*kW] (transposed of our [ksz, outC])
                Tensor wtT = new Tensor(new int[]{outChannels, ksz});
                wtT.toGPU();
                CUDAOps.transpose(wt, wtT);
                
                CUDAOps.conv2dForward(x, wtT, bt, out, inChannels, inH, inW, kernelH, kernelW, outChannels, outH, outW, padH, padW, strideH, strideW);
                
                wtT.close();
            } else {
                // im2col for all batches: colAll[b] is [outH*outW, ksz]
                colAll = new float[batch][];
                for (int b = 0; b < batch; b++) {
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
                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                            val = x.data[b * inSize + (ic * inH * inW + ih * inW + iw)];
                                        }
                                        col[colIdx++] = val;
                                    }
                                }
                            }
                        }
                    }
                    colAll[b] = col;
                }

                for (int b = 0; b < batch; b++) {
                    float[] col = colAll[b];
                    for (int pos = 0; pos < outH * outW; pos++) {
                        for (int oc = 0; oc < outChannels; oc++) {
                            float sum = 0f;
                            int base = pos * ksz;
                            for (int k = 0; k < ksz; k++) {
                                sum += col[base + k] * wt.data[k * outChannels + oc];
                            }
                            out.data[b * outSize + (oc * outH * outW + pos)] = sum + (bt != null ? bt.data[oc] : 0f);
                        }
                    }
                }
            }
            final float[][] colAllFinal = colAll;

            // Autograd backward
            if (Torch.is_grad_enabled() && (x.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x, weight.getTensor(), bias == null ? null : bias.getTensor()) {
                    public void apply(Tensor outGrad) {
                        float[][] localColAll = colAllFinal;
                        if (localColAll == null) {
                            // Recompute im2col if we were on GPU
                            x.toCPU();
                            localColAll = new float[batch][];
                            for (int b = 0; b < batch; b++) {
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
                                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                                        val = x.data[b * inSize + (ic * inH * inW + ih * inW + iw)];
                                                    }
                                                    col[colIdx++] = val;
                                                }
                                            }
                                        }
                                    }
                                }
                                localColAll[b] = col;
                            }
                        }

                        // outGrad: [batch, outC * outH * outW]
                        // Gradient w.r.t. weight: sum_b col[b]^T * dOut[b]
                        if (wt.requires_grad) {
                            Tensor gw = new Tensor(wt.shape);
                            for (int b = 0; b < batch; b++) {
                                float[] col = localColAll[b];
                                // For each (k, oc): gw[k][oc] += sum_pos col[pos*ksz+k] * dOut[b,
                                // oc*outH*outW+pos]
                                for (int pos = 0; pos < outH * outW; pos++) {
                                    for (int oc = 0; oc < outChannels; oc++) {
                                        float dVal = outGrad.data[b * outSize + (oc * outH * outW + pos)];
                                        int colBase = pos * ksz;
                                        for (int k = 0; k < ksz; k++) {
                                            gw.data[k * outChannels + oc] += col[colBase + k] * dVal;
                                        }
                                    }
                                }
                            }
                            wt.backwardStep(gw);
                        }

                        // Gradient w.r.t. bias: sum over batch and spatial
                        if (bt != null && bt.requires_grad) {
                            Tensor gb = new Tensor(bt.shape);
                            for (int b = 0; b < batch; b++) {
                                for (int oc = 0; oc < outChannels; oc++) {
                                    for (int pos = 0; pos < outH * outW; pos++) {
                                        gb.data[oc] += outGrad.data[b * outSize + (oc * outH * outW + pos)];
                                    }
                                }
                            }
                            bt.backwardStep(gb);
                        }

                        // Gradient w.r.t. input: col2im
                        if (x.requires_grad) {
                            Tensor gx = new Tensor(x.shape);
                            for (int b = 0; b < batch; b++) {
                                // dCol[pos*ksz+k] = sum_oc dOut[b, oc*outH*outW+pos] * wt[k, oc]
                                for (int pos = 0; pos < outH * outW; pos++) {
                                    int oh = pos / outW;
                                    int ow = pos % outW;
                                    for (int ic = 0; ic < inChannels; ic++) {
                                        for (int kh = 0; kh < kernelH; kh++) {
                                            for (int kw = 0; kw < kernelW; kw++) {
                                                int ih = oh * strideH - padH + kh;
                                                int iw2 = ow * strideW - padW + kw;
                                                if (ih >= 0 && ih < inH && iw2 >= 0 && iw2 < inW) {
                                                    int kIdx = ic * kernelH * kernelW + kh * kernelW + kw;
                                                    float dColVal = 0f;
                                                    for (int oc = 0; oc < outChannels; oc++) {
                                                        dColVal += outGrad.data[b * outSize + (oc * outH * outW + pos)]
                                                                * wt.data[kIdx * outChannels + oc];
                                                    }
                                                    gx.data[b * inSize + (ic * inH * inW + ih * inW + iw2)] += dColVal;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            x.backwardStep(gx);
                        }
                    }
                };
            }
            return out;
        }

        // convenience constructor: symmetric stride and padding
        public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int stride,
                int padding, NN outer, boolean biasFlag) {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelH = kernelH;
            this.kernelW = kernelW;
            this.inH = inH;
            this.inW = inW;
            this.strideH = stride;
            this.strideW = stride;
            this.padH = padding;
            this.padW = padding;
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

    // --- ConvTranspose2d (transposed / fractionally-strided convolution) ---
    public static class ConvTranspose2d extends Module {
        public int inChannels, outChannels, kernelH, kernelW;
        public int inH, inW;
        public int strideH = 1, strideW = 1;
        public int padH = 0, padW = 0;
        public int outputPadH = 0, outputPadW = 0;
        public Parameter weight; // shape: [inC, outC*kH*kW]
        public Parameter bias; // [outC]

        public ConvTranspose2d(NN outer, int inChannels, int outChannels, int kernelH, int kernelW,
                int inH, int inW, int stride, int padding, int outputPadding, boolean biasFlag) {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelH = kernelH;
            this.kernelW = kernelW;
            this.inH = inH;
            this.inW = inW;
            this.strideH = stride;
            this.strideW = stride;
            this.padH = padding;
            this.padW = padding;
            this.outputPadH = outputPadding;
            this.outputPadW = outputPadding;
            Mat w = outer.mat_alloc(inChannels, outChannels * kernelH * kernelW);
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
        public Tensor forward(Tensor x) {
            int batch = x.shape[0];
            int inSize = inChannels * inH * inW;
            int outH = (inH - 1) * strideH - 2 * padH + kernelH + outputPadH;
            int outW = (inW - 1) * strideW - 2 * padW + kernelW + outputPadW;
            int outSize = outChannels * outH * outW;
            Tensor wt = this.weight.getTensor();
            Tensor bt = this.bias != null ? this.bias.getTensor() : null;
            Tensor out = new Tensor(batch, outSize);
            x.toCPU();
            wt.toCPU();
            for (int b = 0; b < batch; b++) {
                for (int ic = 0; ic < inChannels; ic++) {
                    for (int ih = 0; ih < inH; ih++) {
                        for (int iw = 0; iw < inW; iw++) {
                            float xVal = x.data[b * inSize + ic * inH * inW + ih * inW + iw];
                            for (int oc = 0; oc < outChannels; oc++) {
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int oh = ih * strideH - padH + kh;
                                        int ow = iw * strideW - padW + kw;
                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                            int wIdx = ic * (outChannels * kernelH * kernelW) + oc * kernelH * kernelW
                                                    + kh * kernelW + kw;
                                            out.data[b * outSize + oc * outH * outW + oh * outW + ow] += xVal
                                                    * wt.data[wIdx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (bt != null) {
                    bt.toCPU();
                    for (int oc = 0; oc < outChannels; oc++)
                        for (int pos = 0; pos < outH * outW; pos++)
                            out.data[b * outSize + oc * outH * outW + pos] += bt.data[oc];
                }
            }
            if (x.isGPU()) out.toGPU();
            if (Torch.is_grad_enabled() && (x.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x, weight.getTensor(), bias == null ? null : bias.getTensor()) {
                    public void apply(Tensor outGrad) {
                        if (x.requires_grad) {
                            Tensor gx = new Tensor(x.shape);
                            for (int b = 0; b < batch; b++)
                                for (int ic = 0; ic < inChannels; ic++)
                                    for (int ih = 0; ih < inH; ih++)
                                        for (int iw = 0; iw < inW; iw++) {
                                            float sum = 0f;
                                            for (int oc = 0; oc < outChannels; oc++)
                                                for (int kh = 0; kh < kernelH; kh++)
                                                    for (int kw = 0; kw < kernelW; kw++) {
                                                        int oh = ih * strideH - padH + kh;
                                                        int ow = iw * strideW - padW + kw;
                                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                                            int wIdx = ic * (outChannels * kernelH * kernelW)
                                                                    + oc * kernelH * kernelW + kh * kernelW + kw;
                                                            sum += outGrad.data[b * outSize + oc * outH * outW
                                                                    + oh * outW + ow] * wt.data[wIdx];
                                                        }
                                                    }
                                            gx.data[b * inSize + ic * inH * inW + ih * inW + iw] = sum;
                                        }
                            x.backwardStep(gx);
                        }
                        if (wt.requires_grad) {
                            Tensor gw = new Tensor(wt.shape);
                            for (int b = 0; b < batch; b++)
                                for (int ic = 0; ic < inChannels; ic++)
                                    for (int ih = 0; ih < inH; ih++)
                                        for (int iw = 0; iw < inW; iw++) {
                                            float xVal = x.data[b * inSize + ic * inH * inW + ih * inW + iw];
                                            for (int oc = 0; oc < outChannels; oc++)
                                                for (int kh = 0; kh < kernelH; kh++)
                                                    for (int kw = 0; kw < kernelW; kw++) {
                                                        int oh = ih * strideH - padH + kh;
                                                        int ow = iw * strideW - padW + kw;
                                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                                            int wIdx = ic * (outChannels * kernelH * kernelW)
                                                                    + oc * kernelH * kernelW + kh * kernelW + kw;
                                                            gw.data[wIdx] += xVal * outGrad.data[b * outSize
                                                                    + oc * outH * outW + oh * outW + ow];
                                                        }
                                                    }
                                        }
                            wt.backwardStep(gw);
                        }
                        if (bt != null && bt.requires_grad) {
                            Tensor gb = new Tensor(bt.shape);
                            for (int b = 0; b < batch; b++)
                                for (int oc = 0; oc < outChannels; oc++)
                                    for (int pos = 0; pos < outH * outW; pos++)
                                        gb.data[oc] += outGrad.data[b * outSize + oc * outH * outW + pos];
                            bt.backwardStep(gb);
                        }
                    }
                };
            }
            return out;
        }
    }

    // --- MaxPool2d and AvgPool2d (naive) ---
    public static class MaxPool2d extends Module {
        public int kernelH, kernelW, strideH, strideW, padH, padW;
        public int inC, inH, inW;

        public MaxPool2d(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inC, int inH,
                int inW) {
            this.kernelH = kernelH;
            this.kernelW = kernelW;
            this.strideH = strideH;
            this.strideW = strideW;
            this.padH = padH;
            this.padW = padW;
            this.inC = inC;
            this.inH = inH;
            this.inW = inW;
        }

        @Override
        public Tensor forward(Tensor x) {
            int batch = x.shape[0];
            int inSize = inC * inH * inW;
            int outH = (inH + 2 * padH - kernelH) / strideH + 1;
            int outW = (inW + 2 * padW - kernelW) / strideW + 1;
            int outSize = inC * outH * outW;
            Tensor out = new Tensor(batch, outSize);

            if (x.isGPU()) {
                out.toGPU();
                CUDAOps.maxPool2dForward(x, out, inC, inH, inW, kernelH, kernelW, outH, outW, padH, padW, strideH, strideW);
                // Note: maxIndices won't be filled on GPU path currently, breaking backward if enabled
            } else {
                // Track argmax indices for backward
                int[] maxIndices = new int[batch * outSize];
                for (int b = 0; b < batch; b++) {
                    for (int c = 0; c < inC; c++) {
                        for (int oh = 0; oh < outH; oh++) {
                            for (int ow = 0; ow < outW; ow++) {
                                float maxv = Float.NEGATIVE_INFINITY;
                                int maxIdx = -1;
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw2 = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < inH && iw2 >= 0 && iw2 < inW) {
                                            int idx = b * inSize + (c * inH * inW + ih * inW + iw2);
                                            float v = x.data[idx];
                                            if (v > maxv) {
                                                maxv = v;
                                                maxIdx = idx;
                                            }
                                        }
                                    }
                                }
                                int outIdx = b * outSize + (c * outH * outW + oh * outW + ow);
                                out.data[outIdx] = maxv;
                                maxIndices[outIdx] = maxIdx;
                            }
                        }
                    }
                }
                
                if (Torch.is_grad_enabled() && x.requires_grad) {
                    out.requires_grad = true;
                    out.grad_fn = new Tensor.GradFn(x) {
                        public void apply(Tensor outGrad) {
                            Tensor gx = new Tensor(x.shape);
                            for (int i = 0; i < outGrad.data.length; i++) {
                                if (maxIndices[i] >= 0) {
                                    gx.data[maxIndices[i]] += outGrad.data[i];
                                }
                            }
                            x.backwardStep(gx);
                        }
                    };
                }
            }
            return out;
        }
    }

    public static class AvgPool2d extends Module {
        public int kernelH, kernelW, strideH, strideW, padH, padW;
        public int inC, inH, inW;

        public AvgPool2d(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inC, int inH,
                int inW) {
            this.kernelH = kernelH;
            this.kernelW = kernelW;
            this.strideH = strideH;
            this.strideW = strideW;
            this.padH = padH;
            this.padW = padW;
            this.inC = inC;
            this.inH = inH;
            this.inW = inW;
        }

        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            int batch = x.shape[0];
            int inSize = inC * inH * inW;
            int outH = (inH + 2 * padH - kernelH) / strideH + 1;
            int outW = (inW + 2 * padW - kernelW) / strideW + 1;
            int outSize = inC * outH * outW;
            Tensor out = new Tensor(batch, outSize);
            // Track counts for backward
            int[] counts = new int[batch * outSize];
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < inC; c++) {
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            float sumv = 0f;
                            int cnt = 0;
                            for (int kh = 0; kh < kernelH; kh++) {
                                for (int kw = 0; kw < kernelW; kw++) {
                                    int ih = oh * strideH - padH + kh;
                                    int iw2 = ow * strideW - padW + kw;
                                    if (ih >= 0 && ih < inH && iw2 >= 0 && iw2 < inW) {
                                        sumv += x.data[b * inSize + (c * inH * inW + ih * inW + iw2)];
                                        cnt++;
                                    }
                                }
                            }
                            int outIdx = b * outSize + (c * outH * outW + oh * outW + ow);
                            out.data[outIdx] = cnt > 0 ? sumv / cnt : 0f;
                            counts[outIdx] = cnt;
                        }
                    }
                }
            }
            if (Torch.is_grad_enabled() && x.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x) {
                    public void apply(Tensor outGrad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int c = 0; c < inC; c++) {
                                for (int oh = 0; oh < outH; oh++) {
                                    for (int ow = 0; ow < outW; ow++) {
                                        int outIdx = b * outSize + (c * outH * outW + oh * outW + ow);
                                        int cnt = counts[outIdx];
                                        if (cnt == 0)
                                            continue;
                                        float grad = outGrad.data[outIdx] / cnt;
                                        for (int kh = 0; kh < kernelH; kh++) {
                                            for (int kw = 0; kw < kernelW; kw++) {
                                                int ih = oh * strideH - padH + kh;
                                                int iw2 = ow * strideW - padW + kw;
                                                if (ih >= 0 && ih < inH && iw2 >= 0 && iw2 < inW) {
                                                    gx.data[b * inSize + (c * inH * inW + ih * inW + iw2)] += grad;
                                                }
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
    }

    // --- Zero padding utility ---
    public static class ZeroPad2d extends Module {
        public int padH, padW, inC, inH, inW;

        public ZeroPad2d(int padH, int padW, int inC, int inH, int inW) {
            this.padH = padH;
            this.padW = padW;
            this.inC = inC;
            this.inH = inH;
            this.inW = inW;
        }

        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            int batch = x.shape[0];
            int inSize = inC * inH * inW;
            int outH = inH + 2 * padH;
            int outW = inW + 2 * padW;
            int outSize = inC * outH * outW;
            Tensor out = new Tensor(batch, outSize);
            if (x.isGPU()) out.toGPU();
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < inC; c++) {
                    for (int h = 0; h < inH; h++) {
                        for (int w = 0; w < inW; w++) {
                            float v = x.data[b * inSize + (c * inH * inW + h * inW + w)];
                            int outIdx = b * outSize + (c * outH * outW + (h + padH) * outW + (w + padW));
                            out.data[outIdx] = v;
                        }
                    }
                }
            }
            if (Torch.is_grad_enabled() && x.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x) {
                    public void apply(Tensor outGrad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int c = 0; c < inC; c++) {
                                for (int h = 0; h < inH; h++) {
                                    for (int w = 0; w < inW; w++) {
                                        int outIdx = b * outSize + (c * outH * outW + (h + padH) * outW + (w + padW));
                                        gx.data[b * inSize + (c * inH * inW + h * inW + w)] += outGrad.data[outIdx];
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
    }

    // --- Batch 3: MaxPool1d, AvgPool1d, AdaptiveAvgPool2d ---

    public static class MaxPool1d extends Module {
        public int kernel, stride, pad;

        public MaxPool1d(int kernel, int stride, int pad) {
            this.kernel = kernel;
            this.stride = stride;
            this.pad = pad;
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.max_pool1d(x, kernel, stride, pad);
        }
    }

    public static class AvgPool1d extends Module {
        public int kernel, stride, pad;

        public AvgPool1d(int kernel, int stride, int pad) {
            this.kernel = kernel;
            this.stride = stride;
            this.pad = pad;
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.avg_pool1d(x, kernel, stride, pad);
        }
    }

    public static class AdaptiveAvgPool2d extends Module {
        public int outputH, outputW;

        public AdaptiveAvgPool2d(int outputH, int outputW) {
            this.outputH = outputH;
            this.outputW = outputW;
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.adaptive_avg_pool2d(x, new int[] { outputH, outputW });
        }
    }

    // --- Batch 4: Conv1d, Bilinear, GroupNorm ---

    public static class Conv1d extends Module {
        public int inC, outC, kernel, stride, pad;
        public Parameter weight;
        public Parameter bias;

        public Conv1d(NN outer, int inC, int outC, int kernel, int stride, int pad, boolean useBias) {
            this.inC = inC;
            this.outC = outC;
            this.kernel = kernel;
            this.stride = stride;
            this.pad = pad;
            Mat w = outer.mat_alloc(outC, inC * kernel);
            outer.mat_rand(w, -0.1f, 0.1f);
            this.weight = new Parameter(new Tensor(w.es, outC, inC, kernel)); 
            addParameter("weight", this.weight);
            if (useBias) {
                Mat b = outer.mat_alloc(1, outC);
                outer.mat_fill(b, 0f);
                this.bias = new Parameter(b);
                addParameter("bias", this.bias);
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            return Torch.conv1d(x, weight.getTensor(), bias != null ? bias.getTensor() : null, stride, pad);
        }
    }

    public static class Bilinear extends Module {
        public int d1, d2, outC;
        public Parameter weight;
        public Parameter bias;

        public Bilinear(NN outer, int d1, int d2, int outC, boolean useBias) {
            this.d1 = d1;
            this.d2 = d2;
            this.outC = outC;
            Mat w = outer.mat_alloc(outC, d1 * d2);
            outer.mat_rand(w, -0.1f, 0.1f);
            this.weight = new Parameter(new Tensor(w.es, outC, d1, d2));
            addParameter("weight", this.weight);
            if (useBias) {
                Mat b = outer.mat_alloc(1, outC);
                outer.mat_fill(b, 0f);
                this.bias = new Parameter(b);
                addParameter("bias", this.bias);
            }
        }

        public Tensor forward(Tensor x1, Tensor x2) {
            return Torch.bilinear(x1, x2, weight.getTensor(), bias != null ? bias.getTensor() : null);
        }
    }

    public static class GroupNorm extends Module {
        public int numGroups;
        public int numChannels;
        public float eps;
        public Parameter weight;
        public Parameter bias;

        public GroupNorm(NN outer, int numGroups, int numChannels, float eps) {
            this.numGroups = numGroups;
            this.numChannels = numChannels;
            this.eps = eps;
            Mat w = outer.mat_alloc(1, numChannels);
            outer.mat_fill(w, 1.0f);
            this.weight = new Parameter(w);
            addParameter("weight", this.weight);
            Mat b = outer.mat_alloc(1, numChannels);
            outer.mat_fill(b, 0.0f);
            this.bias = new Parameter(b);
            addParameter("bias", this.bias);
        }

        public GroupNorm(NN outer, int numGroups, int numChannels) {
            this(outer, numGroups, numChannels, 1e-5f);
        }

        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            int n = x.shape[0];
            int c = numChannels;
            int g = numGroups;
            if (c % g != 0) throw new IllegalArgumentException("channels must be divisible by groups");
            int cpG = c / g;
            
            // Flatten spatial dims
            int spatial = 1;
            for (int i = 2; i < x.shape.length; i++) spatial *= x.shape[i];
            
            Tensor out = new Tensor(x.shape);
            if (x.isGPU()) out.toGPU();
            Tensor wt = weight.getTensor();
            wt.toCPU();
            Tensor bt = bias.getTensor();
            bt.toCPU();
            
            final float[] groupMeans = new float[n * g];
            final float[] groupVars = new float[n * g];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < g; j++) {
                    float sum = 0;
                    for (int k = 0; k < cpG; k++) {
                        for (int s = 0; s < spatial; s++) {
                            sum += x.data[i * c * spatial + (j * cpG + k) * spatial + s];
                        }
                    }
                    float mean = sum / (cpG * spatial);
                    groupMeans[i * g + j] = mean;
                    
                    float vsum = 0;
                    for (int k = 0; k < cpG; k++) {
                        for (int s = 0; s < spatial; s++) {
                            float diff = x.data[i * c * spatial + (j * cpG + k) * spatial + s] - mean;
                            vsum += diff * diff;
                        }
                    }
                    float var = vsum / (cpG * spatial);
                    groupVars[i * g + j] = var;
                    
                    float invStd = 1.0f / (float)Math.sqrt(var + eps);
                    for (int k = 0; k < cpG; k++) {
                        int ch = j * cpG + k;
                        for (int s = 0; s < spatial; s++) {
                            int idx = i * c * spatial + ch * spatial + s;
                            float norm = (x.data[idx] - mean) * invStd;
                            out.data[idx] = norm * wt.data[ch] + bt.data[ch];
                        }
                    }
                }
            }

            if (Torch.is_grad_enabled() && (x.requires_grad || wt.requires_grad || bt.requires_grad)) {
                out.requires_grad = true;
                final int fSpatial = spatial;
                out.grad_fn = new Tensor.GradFn(x, weight.getTensor(), bias.getTensor()) {
                    public void apply(Tensor outGrad) {
                        if (bt.requires_grad) {
                            Tensor gb = new Tensor(bt.shape);
                            for (int i = 0; i < n; i++)
                                for (int ch = 0; ch < c; ch++)
                                    for (int s = 0; s < fSpatial; s++)
                                        gb.data[ch] += outGrad.data[i * c * fSpatial + ch * fSpatial + s];
                            bt.backwardStep(gb);
                        }
                        if (wt.requires_grad) {
                            Tensor gg = new Tensor(wt.shape);
                            for (int i = 0; i < n; i++) {
                                for (int j = 0; j < g; j++) {
                                    float invStd = 1.0f / (float)Math.sqrt(groupVars[i * g + j] + eps);
                                    for (int k = 0; k < cpG; k++) {
                                        int ch = j * cpG + k;
                                        for (int s = 0; s < fSpatial; s++) {
                                            int idx = i * c * fSpatial + ch * fSpatial + s;
                                            float norm = (x.data[idx] - groupMeans[i * g + j]) * invStd;
                                            gg.data[ch] += outGrad.data[idx] * norm;
                                        }
                                    }
                                }
                            }
                            wt.backwardStep(gg);
                        }
                        if (x.requires_grad) {
                            Tensor gx = new Tensor(x.shape);
                            int m = cpG * fSpatial;
                            for (int i = 0; i < n; i++) {
                                for (int j = 0; j < g; j++) {
                                    float var = groupVars[i * g + j];
                                    float invStd = 1.0f / (float)Math.sqrt(var + eps);
                                    
                                    float term1 = 0; // sum(og * gamma)
                                    float term2 = 0; // sum(og * gamma * norm)
                                    for (int k = 0; k < cpG; k++) {
                                        int ch = j * cpG + k;
                                        for (int s = 0; s < fSpatial; s++) {
                                            int idx = i * c * fSpatial + ch * fSpatial + s;
                                            float og = outGrad.data[idx];
                                            float norm = (x.data[idx] - groupMeans[i * g + j]) * invStd;
                                            term1 += og * wt.data[ch];
                                            term2 += og * wt.data[ch] * norm;
                                        }
                                    }
                                    
                                    for (int k = 0; k < cpG; k++) {
                                        int ch = j * cpG + k;
                                        for (int s = 0; s < fSpatial; s++) {
                                            int idx = i * c * fSpatial + ch * fSpatial + s;
                                            float norm = (x.data[idx] - groupMeans[i * g + j]) * invStd;
                                            gx.data[idx] = (wt.data[ch] * outGrad.data[idx] - term1/m - norm * term2/m) * invStd;
                                        }
                                    }
                                }
                            }
                            x.backwardStep(gx);
                        }
                    }
                };
            }
            return out;
        }
    }

    // --- LayerNorm ---
    public static class LayerNorm extends Module {
        public int normalizedSize;
        public float eps;
        public Parameter weight; // gamma
        public Parameter bias; // beta

        public LayerNorm(NN outer, int normalizedSize) {
            this(outer, normalizedSize, 1e-5f);
        }

        public LayerNorm(NN outer, int normalizedSize, float eps) {
            this.normalizedSize = normalizedSize;
            this.eps = eps;
            Mat w = outer.mat_alloc(1, normalizedSize);
            outer.mat_fill(w, 1f); // gamma=1
            this.weight = new Parameter(w);
            addParameter("weight", this.weight);
            Mat b = outer.mat_alloc(1, normalizedSize);
            outer.mat_fill(b, 0f); // beta=0
            this.bias = new Parameter(b);
            addParameter("bias", this.bias);
        }

        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            // x: [batch, normalizedSize]
            int batch = x.shape[0];
            int D = normalizedSize;
            Tensor gamma = this.weight.getTensor();
            gamma.toCPU();
            Tensor beta = this.bias.getTensor();
            beta.toCPU();
 
            float[] means = new float[batch];
            float[] vars = new float[batch];
            Tensor out = new Tensor(batch, D);
            if (x.isGPU()) out.toGPU();

            for (int b = 0; b < batch; b++) {
                float sum = 0f;
                for (int d = 0; d < D; d++)
                    sum += x.data[b * D + d];
                means[b] = sum / D;
                float vsum = 0f;
                for (int d = 0; d < D; d++) {
                    float diff = x.data[b * D + d] - means[b];
                    vsum += diff * diff;
                }
                vars[b] = vsum / D;
                float invStd = 1f / (float) Math.sqrt(vars[b] + eps);
                for (int d = 0; d < D; d++) {
                    float norm = (x.data[b * D + d] - means[b]) * invStd;
                    out.data[b * D + d] = gamma.data[d] * norm + beta.data[d];
                }
            }

            if (Torch.is_grad_enabled() && (x.requires_grad || gamma.requires_grad || beta.requires_grad)) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x, weight.getTensor(), bias.getTensor()) {
                    public void apply(Tensor outGrad) {
                        if (beta.requires_grad) {
                            Tensor gb = new Tensor(beta.shape);
                            for (int b = 0; b < batch; b++)
                                for (int d = 0; d < D; d++)
                                    gb.data[d] += outGrad.data[b * D + d];
                            beta.backwardStep(gb);
                        }
                        if (gamma.requires_grad) {
                            Tensor gg = new Tensor(gamma.shape);
                            for (int b = 0; b < batch; b++) {
                                float invStd = 1f / (float) Math.sqrt(vars[b] + eps);
                                for (int d = 0; d < D; d++) {
                                    float norm = (x.data[b * D + d] - means[b]) * invStd;
                                    gg.data[d] += outGrad.data[b * D + d] * norm;
                                }
                            }
                            gamma.backwardStep(gg);
                        }
                        if (x.requires_grad) {
                            Tensor gx = new Tensor(x.shape);
                            for (int b = 0; b < batch; b++) {
                                float invStd = 1f / (float) Math.sqrt(vars[b] + eps);
                                // dxhat = outGrad * gamma
                                float[] dxhat = new float[D];
                                for (int d = 0; d < D; d++)
                                    dxhat[d] = outGrad.data[b * D + d] * gamma.data[d];
                                // compute sum(dxhat) and sum(dxhat * xhat)
                                float sumDxhat = 0f, sumDxhatXhat = 0f;
                                for (int d = 0; d < D; d++) {
                                    float xhat = (x.data[b * D + d] - means[b]) * invStd;
                                    sumDxhat += dxhat[d];
                                    sumDxhatXhat += dxhat[d] * xhat;
                                }
                                for (int d = 0; d < D; d++) {
                                    float xhat = (x.data[b * D + d] - means[b]) * invStd;
                                    gx.data[b * D + d] = invStd / D * (D * dxhat[d] - sumDxhat - xhat * sumDxhatXhat);
                                }
                            }
                            x.backwardStep(gx);
                        }
                    }
                };
            }
            return out;
        }
    }

    // --- InstanceNorm ---
    public static class InstanceNorm extends Module {
        public int numChannels;
        public int spatialH, spatialW;
        public float eps;

        public InstanceNorm(int numChannels, int spatialH, int spatialW) {
            this(numChannels, spatialH, spatialW, 1e-5f);
        }

        public InstanceNorm(int numChannels, int spatialH, int spatialW, float eps) {
            this.numChannels = numChannels;
            this.spatialH = spatialH;
            this.spatialW = spatialW;
            this.eps = eps;
        }

        @Override
        public Tensor forward(Tensor x) {
            x.toCPU();
            // x: [batch, C*H*W]
            int batch = x.shape[0];
            int C = numChannels;
            int HW = spatialH * spatialW;
            int total = C * HW;
            Tensor out = new Tensor(batch, total);
            if (x.isGPU()) out.toGPU();

            float[][] means = new float[batch][C];
            float[][] vars = new float[batch][C];

            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0f;
                    for (int hw = 0; hw < HW; hw++)
                        sum += x.data[b * total + c * HW + hw];
                    means[b][c] = sum / HW;
                    float vsum = 0f;
                    for (int hw = 0; hw < HW; hw++) {
                        float diff = x.data[b * total + c * HW + hw] - means[b][c];
                        vsum += diff * diff;
                    }
                    vars[b][c] = vsum / HW;
                    float invStd = 1f / (float) Math.sqrt(vars[b][c] + eps);
                    for (int hw = 0; hw < HW; hw++) {
                        out.data[b * total + c * HW + hw] = (x.data[b * total + c * HW + hw] - means[b][c]) * invStd;
                    }
                }
            }

            if (Torch.is_grad_enabled() && x.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x) {
                    public void apply(Tensor outGrad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int b = 0; b < batch; b++) {
                            for (int c = 0; c < C; c++) {
                                float invStd = 1f / (float) Math.sqrt(vars[b][c] + eps);
                                float sumDy = 0f, sumDyXhat = 0f;
                                for (int hw = 0; hw < HW; hw++) {
                                    float dy = outGrad.data[b * total + c * HW + hw];
                                    float xhat = (x.data[b * total + c * HW + hw] - means[b][c]) * invStd;
                                    sumDy += dy;
                                    sumDyXhat += dy * xhat;
                                }
                                for (int hw = 0; hw < HW; hw++) {
                                    float dy = outGrad.data[b * total + c * HW + hw];
                                    float xhat = (x.data[b * total + c * HW + hw] - means[b][c]) * invStd;
                                    gx.data[b * total + c * HW + hw] = invStd / HW
                                            * (HW * dy - sumDy - xhat * sumDyXhat);
                                }
                            }
                        }
                        x.backwardStep(gx);
                    }
                };
            }
            return out;
        }
    }

    // --- RNNCell ---
    public static class RNNCell extends Module {
        public int inputSize;
        public int hiddenSize;
        public Parameter weight_ih;
        public Parameter weight_hh;
        public Parameter bias_ih;
        public Parameter bias_hh;

        public RNNCell(NN outer, int inputSize, int hiddenSize, boolean bias) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;

            // He init
            float k = (float) Math.sqrt(1.0 / hiddenSize);
            Mat w_ih = outer.mat_alloc(inputSize, hiddenSize);
            outer.mat_rand(w_ih, -k, k);
            this.weight_ih = new Parameter(w_ih);
            addParameter("weight_ih", this.weight_ih);

            Mat w_hh = outer.mat_alloc(hiddenSize, hiddenSize);
            outer.mat_rand(w_hh, -k, k);
            this.weight_hh = new Parameter(w_hh);
            addParameter("weight_hh", this.weight_hh);

            if (bias) {
                Mat b_ih = outer.mat_alloc(1, hiddenSize);
                outer.mat_fill(b_ih, 0f);
                this.bias_ih = new Parameter(b_ih);
                addParameter("bias_ih", this.bias_ih);

                Mat b_hh = outer.mat_alloc(1, hiddenSize);
                outer.mat_fill(b_hh, 0f);
                this.bias_hh = new Parameter(b_hh);
                addParameter("bias_hh", this.bias_hh);
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            // Forward pass for one time step
            // x: [batch, inputSize]
            // Need hidden state as argument? PyTorch RNNCell takes (input, hidden)
            // But Module.forward only takes one argument.
            // We can define a custom forward or call it directly.
            throw new UnsupportedOperationException("RNNCell requires (input, hidden). Use forward(x, h).");
        }

        public Tensor forward(Tensor x, Tensor h) {
            // h_next = tanh(x*W_ih + b_ih + h*W_hh + b_hh)
            Tensor x_w = Torch.matmul(x, weight_ih.getTensor());
            if (bias_ih != null)
                x_w = Torch.add(x_w, bias_ih.getTensor());

            Tensor h_w = Torch.matmul(h, weight_hh.getTensor());
            if (bias_hh != null)
                h_w = Torch.add(h_w, bias_hh.getTensor());

            return Torch.tanh(Torch.add(x_w, h_w));
        }
    }

    // --- LSTMCell ---
    public static class LSTMCell extends Module {
        public int inputSize;
        public int hiddenSize;
        public Parameter weight_ih;
        public Parameter weight_hh;
        public Parameter bias_ih;
        public Parameter bias_hh;

        public LSTMCell(NN outer, int inputSize, int hiddenSize, boolean bias) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;

            float k = (float) Math.sqrt(1.0 / hiddenSize);
            // Packed weight for i, f, g, o gates
            Mat w_ih = outer.mat_alloc(inputSize, 4 * hiddenSize);
            outer.mat_rand(w_ih, -k, k);
            this.weight_ih = new Parameter(w_ih);
            addParameter("weight_ih", this.weight_ih);

            Mat w_hh = outer.mat_alloc(hiddenSize, 4 * hiddenSize);
            outer.mat_rand(w_hh, -k, k);
            this.weight_hh = new Parameter(w_hh);
            addParameter("weight_hh", this.weight_hh);

            if (bias) {
                Mat b_ih = outer.mat_alloc(1, 4 * hiddenSize);
                outer.mat_fill(b_ih, 0f);
                this.bias_ih = new Parameter(b_ih);
                addParameter("bias_ih", this.bias_ih);

                Mat b_hh = outer.mat_alloc(1, 4 * hiddenSize);
                outer.mat_fill(b_hh, 0f);
                this.bias_hh = new Parameter(b_hh);
                addParameter("bias_hh", this.bias_hh);
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("LSTMCell requires (input, hidden, cell). Use forward(x, h, c).");
        }

        public Tensor[] forward(Tensor x, Tensor h, Tensor c) {
            // Gates calculation
            Tensor x_w = Torch.matmul(x, weight_ih.getTensor());
            if (bias_ih != null)
                x_w = Torch.add(x_w, bias_ih.getTensor());

            Tensor h_w = Torch.matmul(h, weight_hh.getTensor());
            if (bias_hh != null)
                h_w = Torch.add(h_w, bias_hh.getTensor());

            Tensor gates = Torch.add(x_w, h_w);

            // Split gates: [batch, 4*hiddenSize] -> 4 * [batch, hiddenSize]
            java.util.List<Tensor> splitGates = Torch.chunk(gates, 4, 1);
            Tensor i_t = Torch.sigmoid(splitGates.get(0));
            Tensor f_t = Torch.sigmoid(splitGates.get(1));
            Tensor g_t = Torch.tanh(splitGates.get(2));
            Tensor o_t = Torch.sigmoid(splitGates.get(3));

            // c_next = f_t * c + i_t * g_t
            Tensor c_next = Torch.add(Torch.mul(f_t, c), Torch.mul(i_t, g_t));
            // h_next = o_t * tanh(c_next)
            Tensor h_next = Torch.mul(o_t, Torch.tanh(c_next));

            return new Tensor[] { h_next, c_next };
        }
    }

    // --- RNN ---
    public static class RNN extends Module {
        public RNNCell cell;
        public boolean batchFirst;

        public RNN(NN outer, int inputSize, int hiddenSize, boolean bias, boolean batchFirst) {
            this.cell = new RNNCell(outer, inputSize, hiddenSize, bias);
            this.batchFirst = batchFirst;
            addModule("cell", this.cell);
        }

        @Override
        public Tensor forward(Tensor x) {
            // Default forward: input [seq_len, batch, input_size] or [batch, seq_len,
            // input_size]
            // We assume batch_first can be toggled
            int seqLen = batchFirst ? x.shape[1] : x.shape[0];
            int batch = batchFirst ? x.shape[0] : x.shape[1];
            int inputSize = x.shape[2];
            int hiddenSize = cell.hiddenSize;

            Tensor h = Torch.zeros(batch, hiddenSize).to(x.device);

            // Collect outputs
            java.util.List<Tensor> outputs = new java.util.ArrayList<>();
            for (int t = 0; t < seqLen; t++) {
                Tensor xt;
                if (batchFirst) {
                    // Extract [batch, 1, input_size] -> [batch, input_size]
                    xt = Torch.reshape(Torch.narrow(x, 1, t, 1), batch, inputSize);
                } else {
                    xt = Torch.reshape(Torch.narrow(x, 0, t, 1), batch, inputSize);
                }
                h = cell.forward(xt, h);
                outputs.add(h);
            }

            // Stack outputs along dim 0 or 1
            return Torch.stack(outputs, batchFirst ? 1 : 0);
        }
    }

    // --- LSTM ---
    public static class LSTM extends Module {
        public LSTMCell cell;
        public boolean batchFirst;

        public LSTM(NN outer, int inputSize, int hiddenSize, boolean bias, boolean batchFirst) {
            this.cell = new LSTMCell(outer, inputSize, hiddenSize, bias);
            this.batchFirst = batchFirst;
            addModule("cell", this.cell);
        }

        @Override
        public Tensor forward(Tensor x) {
            int seqLen = batchFirst ? x.shape[1] : x.shape[0];
            int batch = batchFirst ? x.shape[0] : x.shape[1];
            int inputSize = x.shape[2];
            int hiddenSize = cell.hiddenSize;

            Tensor h = Torch.zeros(batch, hiddenSize).to(x.device);
            Tensor c = Torch.zeros(batch, hiddenSize).to(x.device);

            java.util.List<Tensor> outputs = new java.util.ArrayList<>();
            for (int t = 0; t < seqLen; t++) {
                Tensor xt;
                if (batchFirst) {
                    xt = Torch.reshape(Torch.narrow(x, 1, t, 1), batch, inputSize);
                } else {
                    xt = Torch.reshape(Torch.narrow(x, 0, t, 1), batch, inputSize);
                }
                Tensor[] nc = cell.forward(xt, h, c);
                h = nc[0];
                c = nc[1];
                outputs.add(h);
            }
            return Torch.stack(outputs, batchFirst ? 1 : 0);
        }
    }

    // --- GRUCell ---
    public static class GRUCell extends Module {
        public int inputSize, hiddenSize;
        public Parameter weight_ih, weight_hh, bias_ih, bias_hh;

        public GRUCell(NN outer, int inputSize, int hiddenSize, boolean bias) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            float k = (float) Math.sqrt(1.0 / hiddenSize);
            Mat w_ih = outer.mat_alloc(inputSize, 3 * hiddenSize);
            outer.mat_rand(w_ih, -k, k);
            this.weight_ih = new Parameter(w_ih);
            addParameter("weight_ih", this.weight_ih);
            Mat w_hh = outer.mat_alloc(hiddenSize, 3 * hiddenSize);
            outer.mat_rand(w_hh, -k, k);
            this.weight_hh = new Parameter(w_hh);
            addParameter("weight_hh", this.weight_hh);
            if (bias) {
                Mat b_ih = outer.mat_alloc(1, 3 * hiddenSize);
                outer.mat_fill(b_ih, 0f);
                this.bias_ih = new Parameter(b_ih);
                addParameter("bias_ih", this.bias_ih);
                Mat b_hh = outer.mat_alloc(1, 3 * hiddenSize);
                outer.mat_fill(b_hh, 0f);
                this.bias_hh = new Parameter(b_hh);
                addParameter("bias_hh", this.bias_hh);
            }
        }

        @Override public Tensor forward(Tensor x) { throw new UnsupportedOperationException("Forward(x, h) required"); }

        public Tensor forward(Tensor x, Tensor h) {
            Tensor x_w = Torch.matmul(x, weight_ih.getTensor());
            if (bias_ih != null) x_w = Torch.add(x_w, bias_ih.getTensor());
            Tensor h_w = Torch.matmul(h, weight_hh.getTensor());
            if (bias_hh != null) h_w = Torch.add(h_w, bias_hh.getTensor());
            java.util.List<Tensor> x_g = Torch.chunk(x_w, 3, 1);
            java.util.List<Tensor> h_g = Torch.chunk(h_w, 3, 1);
            Tensor r = Torch.sigmoid(Torch.add(x_g.get(0), h_g.get(0)));
            Tensor z = Torch.sigmoid(Torch.add(x_g.get(1), h_g.get(1)));
            Tensor n = Torch.tanh(Torch.add(x_g.get(2), Torch.mul(r, h_g.get(2))));
            return Torch.add(Torch.mul(Torch.sub(1.0f, z), n), Torch.mul(z, h));
        }
    }

    // --- GRU ---
    public static class GRU extends Module {
        public GRUCell cell;
        public boolean batchFirst;
        public GRU(NN outer, int inputSize, int hiddenSize, boolean bias, boolean batchFirst) {
            this.cell = new GRUCell(outer, inputSize, hiddenSize, bias);
            this.batchFirst = batchFirst;
            addModule("cell", this.cell);
        }
        @Override public Tensor forward(Tensor x) {
            int seqLen = batchFirst ? x.shape[1] : x.shape[0];
            int batch = batchFirst ? x.shape[0] : x.shape[1];
            int inputSize = x.shape[2];
            Tensor h = Torch.zeros(batch, cell.hiddenSize).to(x.device);
            java.util.List<Tensor> outputs = new java.util.ArrayList<>();
            for (int t = 0; t < seqLen; t++) {
                Tensor xt = batchFirst ? Torch.reshape(Torch.narrow(x, 1, t, 1), batch, inputSize)
                                     : Torch.reshape(Torch.narrow(x, 0, t, 1), batch, inputSize);
                h = cell.forward(xt, h);
                outputs.add(h);
            }
            return Torch.stack(outputs, batchFirst ? 1 : 0);
        }
    }
}
