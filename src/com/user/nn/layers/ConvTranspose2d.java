package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class ConvTranspose2d extends Module {
    public int inChannels, outChannels, kernelH, kernelW;
    public int inH, inW;
    public int strideH = 1, strideW = 1;
    public int padH = 0, padW = 0;
    public int outputPadH = 0, outputPadW = 0;
    public Parameter weight;
    public Parameter bias;

    public ConvTranspose2d(int inChannels, int outChannels, int kernelH, int kernelW,
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
        NN.Mat w = NN.mat_alloc(inChannels, outChannels * kernelH * kernelW);
        this.weight = new Parameter(w);
        Torch.nn.init.kaiming_uniform_(this.weight.getTensor());
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
        int inSize = inChannels * inH * inW;
        int outH = (inH - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outW = (inW - 1) * strideW - 2 * padW + kernelW + outputPadW;
        int outSize = outChannels * outH * outW;
        Tensor wt = this.weight.getTensor();
        Tensor bt = this.bias != null ? this.bias.getTensor() : null;
        Tensor out = new Tensor(batch, outChannels, outH, outW);
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
