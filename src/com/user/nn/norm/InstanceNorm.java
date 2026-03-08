package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class InstanceNorm extends Module {
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
