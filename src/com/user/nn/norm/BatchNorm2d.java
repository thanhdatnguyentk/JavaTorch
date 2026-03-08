package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class BatchNorm2d extends Module {
    public int numFeatures;
    public Parameter weight;
    public Parameter bias;
    public float[] runningMean;
    public float[] runningVar;
    public float eps = 1e-5f;
    public float momentum = 0.1f;

    public BatchNorm2d(int numFeatures) {
        this(numFeatures, true);
    }

    public BatchNorm2d(int numFeatures, boolean affine) {
        this.numFeatures = numFeatures;
        runningMean = new float[numFeatures];
        runningVar = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            runningMean[i] = 0f;
            runningVar[i] = 1f;
        }
        if (affine) {
            NN.Mat gw = NN.mat_alloc(1, numFeatures);
            NN.mat_fill(gw, 1.0f);
            weight = new Parameter(gw);
            addParameter("weight", weight);
            NN.Mat gb = NN.mat_alloc(1, numFeatures);
            NN.mat_fill(gb, 0.0f);
            bias = new Parameter(gb);
            addParameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        if (x.shape.length != 4) throw new IllegalArgumentException("BatchNorm2d requires 4D tensor [batch, C, H, W], got " + java.util.Arrays.toString(x.shape));
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
        if (wasGPU) x.toCPU();

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
            final int outH = x.shape[2], outW = x.shape[3];
            out.grad_fn = new Tensor.GradFn(x, weight != null ? weight.getTensor() : null, bias != null ? bias.getTensor() : null) {
                public void apply(Tensor outGrad) {
                    if (outGrad.isGPU()) outGrad.toCPU();
                    if (x.isGPU()) x.toCPU();
                    Tensor gx = new Tensor(x.shape);
                    Tensor wt = weight != null ? weight.getTensor() : null;
                    if (wt != null && wt.isGPU()) wt.toCPU();
                    Tensor bt = bias != null ? bias.getTensor() : null;
                    if (bt != null && bt.isGPU()) bt.toCPU();
                    
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
            x.toGPU();
            out.toGPU();
        }
        return out;
    }
}
