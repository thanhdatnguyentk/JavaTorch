package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Conv2d extends Module {
    public int inChannels, outChannels, kernelH, kernelW;
    public int strideH, strideW;
    public int padH, padW;
    public Parameter weight;
    public Parameter bias;

    public Conv2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, true);
    }

    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding, boolean biasFlag) {
        this(inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, biasFlag);
    }

    public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, boolean biasFlag) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.strideH = strideH;
        this.strideW = strideW;
        this.padH = padH;
        this.padW = padW;

        // Weight shape: [outChannels, inChannels, kernelH, kernelW]
        Tensor w = new Tensor(outChannels, inChannels, kernelH, kernelW);
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
        // Note: This is an approximation. In a real scenario, H and W should be known.
        // For now, we assume the input is already in 4D or can be reshaped if we have enough info.
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
            throw new IllegalArgumentException("Conv2d expects 4D input [N, C, H, W], got " + x.shape.length + "D");
        }

        int batch = x.shape[0];
        int inC = x.shape[1];
        int inH = x.shape[2];
        int inW = x.shape[3];

        if (inC != inChannels) {
            throw new IllegalArgumentException("Input channels mismatch: expected " + inChannels + " got " + inC);
        }

        int outH = (inH + 2 * padH - kernelH) / strideH + 1;
        int outW = (inW + 2 * padW - kernelW) / strideW + 1;

        Tensor wt = this.weight.getTensor();
        Tensor bt = (this.bias != null) ? this.bias.getTensor() : null;

        final Tensor finalX = x;
        System.out.println("[DEBUG] Conv2d forward: x.isGPU=" + finalX.isGPU() + " wt.isGPU=" + wt.isGPU());
        if (finalX.isGPU()) {
            Tensor out = new Tensor(batch, outChannels, outH, outW);
            out.toGPU();
            CUDAOps.conv2dForward(finalX, wt, bt, out, 
                inChannels, inH, inW, 
                kernelH, kernelW, 
                outChannels, outH, outW, 
                padH, padW, strideH, strideW);

            if (Torch.is_grad_enabled() && (finalX.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(finalX, wt, bt) {
                    @Override
                    public void apply(Tensor gradOutput) {
                        if (!gradOutput.isGPU()) gradOutput.toGPU();
                        if (wt.requires_grad) {
                            System.out.println("[DEBUG] Conv2d GPU backward: computing weight grad");
                            Tensor dw = new Tensor(wt.shape).toGPU();
                            CUDAOps.conv2dBackwardFilter(finalX, gradOutput, dw,
                                inChannels, inH, inW, kernelH, kernelW,
                                outChannels, outH, outW, padH, padW, strideH, strideW);
                            wt.backwardStep(dw);
                        }
                        if (bt != null && bt.requires_grad) {
                            Tensor db = new Tensor(bt.shape).toGPU();
                            CUDAOps.conv2dBackwardBias(gradOutput, db, outChannels, outH, outW);
                            bt.backwardStep(db);
                        }
                        if (finalX.requires_grad) {
                            Tensor dx = new Tensor(finalX.shape).toGPU();
                            CUDAOps.conv2dBackwardData(gradOutput, wt, dx,
                                inChannels, inH, inW, kernelH, kernelW,
                                outChannels, outH, outW, padH, padW, strideH, strideW);
                            finalX.backwardStep(dx);
                        }
                    }
                };
            }
            return out;
        } else {
            // CPU Path (Reference implementation)
            Tensor out = new Tensor(batch, outChannels, outH, outW);
            float[] xData = finalX.data;
            float[] wData = wt.data;
            float[] bData = (bt != null) ? bt.data : null;
            float[] outData = out.data;

            for (int b = 0; b < batch; b++) {
                for (int oc = 0; oc < outChannels; oc++) {
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            float sum = 0;
                            for (int ic = 0; ic < inChannels; ic++) {
                                for (int kh = 0; kh < kernelH; kh++) {
                                    for (int kw = 0; kw < kernelW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                            float val = xData[((b * inChannels + ic) * inH + ih) * inW + iw];
                                            float weightVal = wData[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                                            sum += val * weightVal;
                                        }
                                    }
                                }
                            }
                            if (bData != null) sum += bData[oc];
                            outData[((b * outChannels + oc) * outH + oh) * outW + ow] = sum;
                        }
                    }
                }
            }

            if (Torch.is_grad_enabled() && (finalX.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(finalX, wt, bt) {
                    @Override
                    public void apply(Tensor gradOutput) {
                        if (wt.requires_grad) {
                            Tensor dw = new Tensor(wt.shape);
                            for (int b = 0; b < batch; b++) {
                                for (int oc = 0; oc < outChannels; oc++) {
                                    for (int oh = 0; oh < outH; oh++) {
                                        for (int ow = 0; ow < outW; ow++) {
                                            float go = gradOutput.data[((b * outChannels + oc) * outH + oh) * outW + ow];
                                            for (int ic = 0; ic < inChannels; ic++) {
                                                for (int kh = 0; kh < kernelH; kh++) {
                                                    for (int kw = 0; kw < kernelW; kw++) {
                                                        int ih = oh * strideH - padH + kh;
                                                        int iw = ow * strideW - padW + kw;
                                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                                            float val = finalX.data[((b * inChannels + ic) * inH + ih) * inW + iw];
                                                            dw.data[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw] += val * go;
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
                        if (finalX.requires_grad) {
                            Tensor dx = new Tensor(finalX.shape);
                            for (int b = 0; b < batch; b++) {
                                for (int oc = 0; oc < outChannels; oc++) {
                                    for (int oh = 0; oh < outH; oh++) {
                                        for (int ow = 0; ow < outW; ow++) {
                                            float go = gradOutput.data[((b * outChannels + oc) * outH + oh) * outW + ow];
                                            for (int ic = 0; ic < inChannels; ic++) {
                                                for (int kh = 0; kh < kernelH; kh++) {
                                                    for (int kw = 0; kw < kernelW; kw++) {
                                                        int ih = oh * strideH - padH + kh;
                                                        int iw = ow * strideW - padW + kw;
                                                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                                            float weightVal = wt.data[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
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
                    }
                };
            }
            return out;
        }
    }
}
