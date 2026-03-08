package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class GroupNorm extends Module {
    public int numGroups;
    public int numChannels;
    public float eps;
    public Parameter weight;
    public Parameter bias;

    public GroupNorm(int numGroups, int numChannels, float eps) {
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.eps = eps;
        NN.Mat w = NN.mat_alloc(1, numChannels);
        NN.mat_fill(w, 1.0f);
        this.weight = new Parameter(w);
        addParameter("weight", this.weight);
        NN.Mat b = NN.mat_alloc(1, numChannels);
        NN.mat_fill(b, 0.0f);
        this.bias = new Parameter(b);
        addParameter("bias", this.bias);
    }

    public GroupNorm(int numGroups, int numChannels) {
        this(numGroups, numChannels, 1e-5f);
    }

    @Override
    public Tensor forward(Tensor x) {
        x.toCPU();
        int n = x.shape[0];
        int c = numChannels;
        int g = numGroups;
        if (c % g != 0) throw new IllegalArgumentException("channels must be divisible by groups");
        int cpG = c / g;
        
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
                                
                                float term1 = 0;
                                float term2 = 0;
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
