package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.activations.ReLU;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

public class Conv2d extends Module {
    public int inChannels, outChannels, kernelH, kernelW;
    public int inH, inW;
    public int strideH = 1, strideW = 1;
    public int padH = 0, padW = 0;
    public Parameter weight;
    public Parameter bias;

    public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int stride,
            int padding, boolean biasFlag) {
        this(inChannels, outChannels, kernelH, kernelW, inH, inW, stride, stride, padding, padding, biasFlag);
    }

    public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int inH, int inW, int strideH,
            int strideW, int padH, int padW, boolean biasFlag) {
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
        NN.Mat w = NN.mat_alloc(ksz, outChannels);
        NN.mat_rand(w, -0.08f, 0.08f);
        this.weight = new Parameter(w);
        addParameter("weight", this.weight);
        if (biasFlag) {
            NN.Mat b = NN.mat_alloc(1, outChannels);
            NN.mat_fill(b, 0f);
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
        Tensor out = new Tensor(batch, outChannels, outH, outW);
        float[][] colAll = null;

        if (x.isGPU()) {
            out.toGPU();
            Tensor wtT = new Tensor(new int[]{outChannels, ksz});
            wtT.toGPU();
            CUDAOps.transpose(wt, wtT);
            
            CUDAOps.conv2dForward(x, wtT, bt, out, inChannels, inH, inW, kernelH, kernelW, outChannels, outH, outW, padH, padW, strideH, strideW);
            
            wtT.close();
        } else {
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
        if (Torch.is_grad_enabled() && (x.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
            out.requires_grad = true;
            final boolean wasGPU = x.isGPU();
            out.grad_fn = new Tensor.GradFn(x, weight.getTensor(), bias == null ? null : bias.getTensor()) {
                public void apply(Tensor outGrad) {
                    if (wasGPU) {
                        if (!outGrad.isGPU()) outGrad.toGPU();
                        if (!x.isGPU()) x.toGPU();
                        if (!wt.isGPU()) wt.toGPU();

                        int ksz2 = inChannels * kernelH * kernelW;
                        Tensor wtT = new Tensor(new int[]{outChannels, ksz2});
                        wtT.toGPU();
                        CUDAOps.transpose(wt, wtT);

                        if (wt.requires_grad) {
                            Tensor dwCudnn = new Tensor(new int[]{outChannels, inChannels, kernelH, kernelW});
                            dwCudnn.toGPU();
                            
                            CUDAOps.conv2dBackwardFilter(x, outGrad, dwCudnn,
                                inChannels, inH, inW, kernelH, kernelW,
                                outChannels, outH, outW, padH, padW, strideH, strideW);
                            
                            Tensor dwFlat = new Tensor(new int[]{outChannels, ksz2});
                            dwFlat.toGPU();
                            cudaMemcpy(dwFlat.getDevicePointer(), dwCudnn.getDevicePointer(),
                                (long) outChannels * ksz2 * jcuda.Sizeof.FLOAT, cudaMemcpyDeviceToDevice);
                            dwFlat.markDirtyOnGPU();
                            
                            Tensor gw = new Tensor(wt.shape);
                            gw.toGPU();
                            CUDAOps.transpose(dwFlat, gw);
                            
                            dwCudnn.close();
                            dwFlat.close();
                            wt.backwardStep(gw);
                        }

                        if (bt != null && bt.requires_grad) {
                            Tensor gb = new Tensor(new int[]{outChannels});
                            gb.toGPU();
                            CUDAOps.conv2dBackwardBias(outGrad, gb, outChannels, outH, outW);
                            bt.backwardStep(gb);
                        }

                        if (x.requires_grad) {
                            Tensor gx = new Tensor(x.shape);
                            gx.toGPU();
                            CUDAOps.conv2dBackwardData(outGrad, wtT, gx,
                                inChannels, inH, inW, kernelH, kernelW,
                                outChannels, outH, outW, padH, padW, strideH, strideW);
                            x.backwardStep(gx);
                        }

                        wtT.close();
                    } else {
                        outGrad.toCPU();
                        x.toCPU();
                        wt.toCPU();

                        float[][] localColAll = new float[batch][];
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

                        if (wt.requires_grad) {
                            Tensor gw = new Tensor(wt.shape);
                            for (int b = 0; b < batch; b++) {
                                float[] col = localColAll[b];
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

                        if (x.requires_grad) {
                            Tensor gx = new Tensor(x.shape);
                            for (int b = 0; b < batch; b++) {
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
                }
            };
        }
        return out;
    }
}
