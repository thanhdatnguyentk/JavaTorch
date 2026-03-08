package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class LayerNorm extends Module {
    public int normalizedSize;
    public float eps;
    public Parameter weight;
    public Parameter bias;

    public LayerNorm(int normalizedSize) {
        this(normalizedSize, 1e-5f);
    }

    public LayerNorm(int normalizedSize, float eps) {
        this.normalizedSize = normalizedSize;
        this.eps = eps;
        NN.Mat w = NN.mat_alloc(1, normalizedSize);
        NN.mat_fill(w, 1f);
        this.weight = new Parameter(w);
        addParameter("weight", this.weight);
        NN.Mat b = NN.mat_alloc(1, normalizedSize);
        NN.mat_fill(b, 0f);
        this.bias = new Parameter(b);
        addParameter("bias", this.bias);
    }

    @Override
    public Tensor forward(Tensor x) {
        x.toCPU();
        final int[] originalShape = x.shape;
        final int D = normalizedSize;
        final int numel = x.numel();
        final int outer = numel / D;
        final float fEps = eps;
        
        final Tensor gamma = this.weight.getTensor();
        gamma.toCPU();
        final Tensor beta = this.bias.getTensor();
        beta.toCPU();

        final float[] means = new float[outer];
        final float[] vars = new float[outer];
        Tensor out = new Tensor(originalShape);

        for (int i = 0; i < outer; i++) {
            float sum = 0f;
            for (int d = 0; d < D; d++)
                sum += x.data[i * D + d];
            means[i] = sum / D;
            float vsum = 0f;
            for (int d = 0; d < D; d++) {
                float diff = x.data[i * D + d] - means[i];
                vsum += diff * diff;
            }
            vars[i] = vsum / D;
            float invStd = 1f / (float) Math.sqrt(vars[i] + fEps);
            for (int d = 0; d < D; d++) {
                float norm = (x.data[i * D + d] - means[i]) * invStd;
                out.data[i * D + d] = gamma.data[d] * norm + beta.data[d];
            }
        }
        if (x.isGPU()) out.toGPU();

        if (Torch.is_grad_enabled() && (x.requires_grad || gamma.requires_grad || beta.requires_grad)) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x, weight.getTensor(), bias.getTensor()) {
                public void apply(Tensor outGrad) {
                    outGrad.toCPU(); x.toCPU();
                    if (beta.requires_grad) {
                        Tensor gb = new Tensor(beta.shape);
                        for (int i = 0; i < outer; i++)
                            for (int d = 0; d < D; d++)
                                gb.data[d] += outGrad.data[i * D + d];
                        beta.backwardStep(gb);
                    }
                    if (gamma.requires_grad) {
                        Tensor gg = new Tensor(gamma.shape);
                        for (int i = 0; i < outer; i++) {
                            float invStd = 1f / (float) Math.sqrt(vars[i] + fEps);
                            for (int d = 0; d < D; d++) {
                                float norm = (x.data[i * D + d] - means[i]) * invStd;
                                gg.data[d] += outGrad.data[i * D + d] * norm;
                            }
                        }
                        gamma.backwardStep(gg);
                    }
                    if (x.requires_grad) {
                        Tensor gx = new Tensor(x.shape);
                        for (int i = 0; i < outer; i++) {
                            float invStd = 1f / (float) Math.sqrt(vars[i] + fEps);
                            float[] dxhat = new float[D];
                            for (int d = 0; d < D; d++)
                                dxhat[d] = outGrad.data[i * D + d] * gamma.data[d];
                            float sumDxhat = 0f, sumDxhatXhat = 0f;
                            for (int d = 0; d < D; d++) {
                                float xhat = (x.data[i * D + d] - means[i]) * invStd;
                                sumDxhat += dxhat[d];
                                sumDxhatXhat += dxhat[d] * xhat;
                            }
                            for (int d = 0; d < D; d++) {
                                float xhat = (x.data[i * D + d] - means[i]) * invStd;
                                gx.data[i * D + d] = invStd / D * (D * dxhat[d] - sumDxhat - xhat * sumDxhatXhat);
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
