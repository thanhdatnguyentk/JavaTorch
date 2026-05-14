package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class BatchNorm2d extends Module {
    public int numFeatures;
    public float momentum;
    public float epsilon;
    public Parameter gamma;
    public Parameter beta;
    public Tensor runningMean;
    public Tensor runningVar;

    public BatchNorm2d(int numFeatures) {
        this(numFeatures, 0.1f, 1e-5f);
    }

    public BatchNorm2d(int numFeatures, float momentum, float epsilon) {
        this.numFeatures = numFeatures;
        this.momentum = momentum;
        this.epsilon = epsilon;

        Tensor g = new Tensor(numFeatures);
        Torch.nn.init.ones_(g);
        this.gamma = new Parameter(g);
        addParameter("gamma", this.gamma);

        Tensor b = new Tensor(numFeatures);
        Torch.nn.init.zeros_(b);
        this.beta = new Parameter(b);
        addParameter("beta", this.beta);

        this.runningMean = new Tensor(numFeatures);
        Torch.nn.init.zeros_(this.runningMean);
        
        this.runningVar = new Tensor(numFeatures);
        Torch.nn.init.ones_(this.runningVar);
    }

    @Override
    public void toGPU() {
        super.toGPU();
        runningMean.toGPU();
        runningVar.toGPU();
    }

    @Override
    public void toCPU() {
        super.toCPU();
        runningMean.toCPU();
        runningVar.toCPU();
    }

    @Override
    public Tensor forward(Tensor x) {
        if (x.shape.length != 4) {
            throw new IllegalArgumentException(
                "BatchNorm2d expects 4D input [N, C, H, W], got " + x.shape.length + "D");
        }
        if (x.shape[1] != numFeatures) {
            throw new IllegalArgumentException(
                "BatchNorm2d: expected " + numFeatures + " channels, got " + x.shape[1]);
        }

        Tensor g = this.gamma.getTensor();
        Tensor b = this.beta.getTensor();

        // GPU path
        if (x.isGPU() && CUDAOps.isAvailable()) {
            Tensor out = new Tensor(x.shape);
            out.toGPU();

            if (this.is_training()) {
                CUDAOps.batchNorm2dForwardTraining(x, out, g, b, runningMean, runningVar, momentum, epsilon);
            } else {
                CUDAOps.batchNorm2dForwardInference(x, out, g, b, runningMean, runningVar, epsilon);
            }

            if (Torch.is_grad_enabled() && (x.requires_grad || g.requires_grad || b.requires_grad)) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x, g, b) {
                    @Override
                    public void apply(Tensor gradOutput) {
                        if (!gradOutput.isGPU()) gradOutput.toGPU();

                        Tensor dx = new Tensor(x.shape);
                        dx.toGPU();

                        Tensor dg = new Tensor(g.shape);
                        dg.toGPU();

                        Tensor db = new Tensor(b.shape);
                        db.toGPU();

                        CUDAOps.batchNorm2dBackward(x, gradOutput, dx, g, dg, db, epsilon);

                        x.backwardStep(dx);
                        g.backwardStep(dg);
                        b.backwardStep(db);
                    }
                };
            }

            return out;
        }

        // CPU fallback path
        x.toCPU();
        g.toCPU();
        b.toCPU();
        runningMean.toCPU();
        runningVar.toCPU();

        int N = x.shape[0];
        int C = x.shape[1];
        int H = x.shape[2];
        int W = x.shape[3];
        int spatialSize = H * W;

        Tensor out = new Tensor(x.shape);

        // Per-channel mean and variance
        float[] mean = new float[C];
        float[] var = new float[C];

        if (this.is_training()) {
            // Compute batch statistics: mean over (N, H, W) for each channel
            int count = N * spatialSize;
            for (int c = 0; c < C; c++) {
                float sum = 0f;
                for (int n = 0; n < N; n++) {
                    int baseIdx = ((n * C) + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        sum += x.data[baseIdx + s];
                    }
                }
                mean[c] = sum / count;
            }
            for (int c = 0; c < C; c++) {
                float sumSq = 0f;
                for (int n = 0; n < N; n++) {
                    int baseIdx = ((n * C) + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float d = x.data[baseIdx + s] - mean[c];
                        sumSq += d * d;
                    }
                }
                var[c] = sumSq / count;
            }
            // Update running statistics
            for (int c = 0; c < C; c++) {
                runningMean.data[c] = (1f - momentum) * runningMean.data[c] + momentum * mean[c];
                runningVar.data[c] = (1f - momentum) * runningVar.data[c] + momentum * var[c];
            }
        } else {
            // Use running stats for inference
            System.arraycopy(runningMean.data, 0, mean, 0, C);
            System.arraycopy(runningVar.data, 0, var, 0, C);
        }

        // Normalize: y = gamma * (x - mean) / sqrt(var + eps) + beta
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float invStd = 1.0f / (float) Math.sqrt(var[c] + epsilon);
                float gVal = g.data[c];
                float bVal = b.data[c];
                int baseIdx = ((n * C) + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    float normed = (x.data[baseIdx + s] - mean[c]) * invStd;
                    out.data[baseIdx + s] = gVal * normed + bVal;
                }
            }
        }

        // Autograd support for CPU path
        if (Torch.is_grad_enabled() && (x.requires_grad || g.requires_grad || b.requires_grad)) {
            final float[] savedMean = mean;
            final float[] savedVar = var;
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x, g, b) {
                @Override
                public void apply(Tensor gradOutput) {
                    gradOutput.toCPU();
                    int count = N * spatialSize;
                    Tensor dx = new Tensor(x.shape);
                    Tensor dg = new Tensor(g.shape);
                    Tensor db = new Tensor(b.shape);

                    for (int c = 0; c < C; c++) {
                        float invStd = 1.0f / (float) Math.sqrt(savedVar[c] + epsilon);

                        // Accumulate dGamma and dBeta
                        float dgSum = 0f;
                        float dbSum = 0f;
                        for (int n1 = 0; n1 < N; n1++) {
                            int base1 = ((n1 * C) + c) * spatialSize;
                            for (int s1 = 0; s1 < spatialSize; s1++) {
                                float xHat = (x.data[base1 + s1] - savedMean[c]) * invStd;
                                dgSum += gradOutput.data[base1 + s1] * xHat;
                                dbSum += gradOutput.data[base1 + s1];
                            }
                        }
                        dg.data[c] = dgSum;
                        db.data[c] = dbSum;

                        // Compute dX
                        float gVal = g.data[c];
                        for (int n2 = 0; n2 < N; n2++) {
                            int base2 = ((n2 * C) + c) * spatialSize;
                            for (int s2 = 0; s2 < spatialSize; s2++) {
                                float xHat = (x.data[base2 + s2] - savedMean[c]) * invStd;
                                float dxHat = gradOutput.data[base2 + s2] * gVal;
                                // dX = (1/count) * invStd * (count * dxHat - dbSum_g - xHat * dgSum_g)
                                dx.data[base2 + s2] = invStd / count *
                                    (count * dxHat - dbSum * gVal - xHat * dgSum * gVal);
                            }
                        }
                    }

                    if (x.requires_grad) x.backwardStep(dx);
                    if (g.requires_grad) g.backwardStep(dg);
                    if (b.requires_grad) b.backwardStep(db);
                }
            };
        }

        return out;
    }
}
