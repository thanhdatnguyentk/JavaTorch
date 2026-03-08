package com.user.nn.core;

public class Functional {
    public static NN.Mat relu(NN.Mat x) {
        Tensor t = Torch.fromMat(x);
        Tensor out = Torch.relu(t);
        NN.Mat m = NN.mat_alloc(x.rows, x.cols);
        System.arraycopy(out.data, 0, m.es, 0, m.es.length);
        return m;
    }

    public static float mse_loss(NN.Mat pred, NN.Mat target) {
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

    public static float cross_entropy_logits(NN.Mat logits, int[] targets) {
        if (logits.rows != targets.length)
            throw new IllegalArgumentException("cross_entropy: batch size mismatch");
        int batch = logits.rows;
        int classes = logits.cols;
        float total = 0f;
        for (int i = 0; i < batch; i++) {
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

    public static Tensor cross_entropy_tensor(Tensor logits, int[] targets) {
        logits.toCPU();
        int batch = logits.shape[0];
        int classes = logits.shape[1];
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

    public static Tensor mse_loss_tensor(Tensor pred, Tensor target) {
        Tensor diff = Torch.sub(pred, target);
        Tensor sq = Torch.mul(diff, diff);
        return Torch.mean_tensor(sq);
    }

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

    public static Tensor binary_cross_entropy(Tensor input, Tensor target) {
        int n = input.data.length;
        Tensor out = new Tensor(1);
        if (input.isGPU() && target.isGPU()) {
            out = new Tensor(1).toGPU();
            Tensor bceOut = new Tensor(input.shape).toGPU();
            CUDAOps.bceForward(input, target, bceOut);
            bceOut.toCPU();
            float total = 0f;
            for (int i = 0; i < n; i++) total += bceOut.data[i];
            out.toCPU();
            out.data[0] = total / n;
            out.toGPU();
            if (input.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(input) {
                    public void apply(Tensor outGrad) {
                        outGrad.toCPU();
                        float scale = outGrad.data[0] / n;
                        input.toCPU(); target.toCPU();
                        Tensor g = new Tensor(input.shape);
                        for (int i = 0; i < n; i++) {
                            float h = Math.max(1e-12f, Math.min(1f - 1e-12f, input.data[i]));
                            float y = target.data[i];
                            g.data[i] = ((h - y) / (h * (1f - h) + 1e-12f)) * scale;
                        }
                        input.toGPU(); target.toGPU(); g.toGPU();
                        input.backwardStep(g);
                    }
                };
            }
            return out;
        }
        input.toCPU();
        target.toCPU();
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

    public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target) {
        int n = input.data.length;
        Tensor out = new Tensor(1);
        if (input.isGPU() && target.isGPU()) {
            out = new Tensor(1).toGPU();
            Tensor bceOut = new Tensor(input.shape).toGPU();
            CUDAOps.bceLogitsForward(input, target, bceOut);
            Tensor s = Torch.sum_tensor(bceOut);
            float val = s.toCPU().data[0] / n;
            out.toCPU();
            out.data[0] = val;
            out.markDirtyOnGPU();
            if (input.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(input) {
                    public void apply(Tensor outGrad) {
                        float scale = outGrad.data[0] / n;
                        Tensor gx = new Tensor(input.shape).toGPU();
                        CUDAOps.bceLogitsBackward(input, target, gx);
                        CUDAOps.mul(gx, scale, gx);
                        input.backwardStep(gx);
                    }
                };
            }
            return out;
        }
        input.toCPU();
        target.toCPU();
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
