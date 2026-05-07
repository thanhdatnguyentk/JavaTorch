package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class ConvTranspose2d extends Module {
    public int inChannels, outChannels, kernelH, kernelW;
    public int strideH, strideW;
    public int padH, padW;
    public int outputPadH, outputPadW;
    public Parameter weight;
    public Parameter bias;

    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, 0, true);
    }

    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, int stride, int padding, int outputPadding, boolean biasFlag) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelH = kernelSize;
        this.kernelW = kernelSize;
        this.strideH = stride;
        this.strideW = stride;
        this.padH = padding;
        this.padW = padding;
        this.outputPadH = outputPadding;
        this.outputPadW = outputPadding;

        // Weight shape: [inChannels, outChannels, kernelH, kernelW]
        Tensor w = new Tensor(inChannels, outChannels, kernelH, kernelW);
        this.weight = new Parameter(w);
        Torch.nn.init.kaiming_uniform_(this.weight.getTensor());
        addParameter("weight", this.weight);

        if (biasFlag) {
            Tensor b = new Tensor(outChannels);
            Torch.nn.init.zeros_(b);
            this.bias = new Parameter(b);
            addParameter("bias", this.bias);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        // Handle 2D input from legacy tests [Batch, Flat] -> [Batch, C, H, W]
        if (x.shape.length == 2) {
            int batch = x.shape[0];
            int totalSpatial = x.shape[1] / inChannels;
            int h = (int)Math.sqrt(totalSpatial);
            int w = h;
            if (h * w * inChannels == x.shape[1]) {
                x = x.reshape(batch, inChannels, h, w);
            }
        }

        if (x.shape.length != 4) {
            throw new IllegalArgumentException("ConvTranspose2d expects 4D input [N, C, H, W], got " + x.shape.length + "D");
        }

        int batch = x.shape[0];
        int inC = x.shape[1];
        int inH = x.shape[2];
        int inW = x.shape[3];

        if (inC != inChannels) {
            throw new IllegalArgumentException("Input channels mismatch: expected " + inChannels + " got " + inC);
        }

        int outH = (inH - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outW = (inW - 1) * strideW - 2 * padW + kernelW + outputPadW;

        Tensor wt = this.weight.getTensor();
        Tensor bt = (this.bias != null) ? this.bias.getTensor() : null;

        final Tensor finalX = x;
        if (finalX.isGPU()) {
            Tensor out = new Tensor(batch, outChannels, outH, outW);
            out.toGPU();

            // ConvTranspose forward is cuDNN conv backward_data
            CUDAOps.conv2dBackwardData(finalX, wt, out, 
                outChannels, outH, outW, 
                kernelH, kernelW,
                inChannels, inH, inW, 
                padH, padW, strideH, strideW);

            if (bt != null) {
                CUDAOps.add(out, bt.reshape(1, outChannels, 1, 1), out);
            }

            if (Torch.is_grad_enabled() && (finalX.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(finalX, wt, bt) {
                    @Override
                    public void apply(Tensor gradOutput) {
                        if (!gradOutput.isGPU()) gradOutput.toGPU();
                        if (finalX.requires_grad) {
                            Tensor dx = new Tensor(finalX.shape).toGPU();
                            CUDAOps.conv2dForward(gradOutput, wt, null, dx,
                                outChannels, outH, outW, kernelH, kernelW,
                                inChannels, inH, inW, padH, padW, strideH, strideW);
                            finalX.backwardStep(dx);
                        }
                        if (wt.requires_grad) {
                            Tensor dw = new Tensor(wt.shape).toGPU();
                            CUDAOps.conv2dBackwardFilter(gradOutput, finalX, dw,
                                outChannels, outH, outW, kernelH, kernelW,
                                inChannels, inH, inW, padH, padW, strideH, strideW);
                            wt.backwardStep(dw);
                        }
                        if (bt != null && bt.requires_grad) {
                            Tensor db = new Tensor(bt.shape).toGPU();
                            CUDAOps.conv2dBackwardBias(gradOutput, db, outChannels, outH, outW);
                            bt.backwardStep(db);
                        }
                    }
                };
            }
            return out;
        } else {
            // CPU Path
            Tensor out = new Tensor(batch, outChannels, outH, outW);
            float[] xData = finalX.data;
            float[] wData = wt.data;
            float[] bData = (bt != null) ? bt.data : null;
            float[] outData = out.data;

            for (int b = 0; b < batch; b++) {
                for (int oc = 0; oc < outChannels; oc++) {
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            if (bData != null) outData[((b * outChannels + oc) * outH + oh) * outW + ow] = bData[oc];
                        }
                    }
                }
                for (int ic = 0; ic < inChannels; ic++) {
                    for (int ih = 0; ih < inH; ih++) {
                        for (int iw = 0; iw < inW; iw++) {
                            float val = xData[((b * inChannels + ic) * inH + ih) * inW + iw];
                            for (int oc = 0; oc < outChannels; oc++) {
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int oh = ih * strideH - padH + kh;
                                        int ow = iw * strideW - padW + kw;
                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                            float weightVal = wData[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                                            outData[((b * outChannels + oc) * outH + oh) * outW + ow] += val * weightVal;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (Torch.is_grad_enabled() && (finalX.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(finalX, wt, bt) {
                    @Override
                    public void apply(Tensor gradOutput) {
                        if (finalX.requires_grad) {
                            Tensor dx = new Tensor(finalX.shape);
                            for (int b = 0; b < batch; b++) {
                                for (int ic = 0; ic < inChannels; ic++) {
                                    for (int ih = 0; ih < inH; ih++) {
                                        for (int iw = 0; iw < inW; iw++) {
                                            for (int oc = 0; oc < outChannels; oc++) {
                                                for (int kh = 0; kh < kernelH; kh++) {
                                                    for (int kw = 0; kw < kernelW; kw++) {
                                                        int oh = ih * strideH - padH + kh;
                                                        int ow = iw * strideW - padW + kw;
                                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                                            float go = gradOutput.data[((b * outChannels + oc) * outH + oh) * outW + ow];
                                                            float weightVal = wt.data[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                                                            dx.data[((b * inChannels + ic) * inH + ih) * inW + iw] += weightVal * go;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            finalX.backwardStep(dx);
                        }
                        if (wt.requires_grad) {
                            Tensor dw = new Tensor(wt.shape);
                            for (int b = 0; b < batch; b++) {
                                for (int ic = 0; ic < inChannels; ic++) {
                                    for (int ih = 0; ih < inH; ih++) {
                                        for (int iw = 0; iw < inW; iw++) {
                                            float val = finalX.data[((b * inChannels + ic) * inH + ih) * inW + iw];
                                            for (int oc = 0; oc < outChannels; oc++) {
                                                for (int kh = 0; kh < kernelH; kh++) {
                                                    for (int kw = 0; kw < kernelW; kw++) {
                                                        int oh = ih * strideH - padH + kh;
                                                        int ow = iw * strideW - padW + kw;
                                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                                            float go = gradOutput.data[((b * outChannels + oc) * outH + oh) * outW + ow];
                                                            dw.data[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw] += val * go;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            wt.backwardStep(dw);
                        }
                        if (bt != null && bt.requires_grad) {
                            Tensor db = new Tensor(bt.shape);
                            for (int b = 0; b < batch; b++) {
                                for (int oc = 0; oc < outChannels; oc++) {
                                    for (int oh = 0; oh < outH; oh++) {
                                        for (int ow = 0; ow < outW; ow++) {
                                            db.data[oc] += gradOutput.data[((b * outChannels + oc) * outH + oh) * outW + ow];
                                        }
                                    }
                                }
                            }
                            bt.backwardStep(db);
                        }
                    }
                };
            }
            return out;
        }
    }
}
