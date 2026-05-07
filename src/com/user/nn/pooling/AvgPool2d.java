package com.user.nn.pooling;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class AvgPool2d extends Module {
    public int kernelH, kernelW, strideH, strideW, padH, padW;
    public int inC, inH, inW;

    public AvgPool2d(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inC, int inH, int inW) {
        this.kernelH = kernelH; this.kernelW = kernelW;
        this.strideH = strideH; this.strideW = strideW;
        this.padH = padH; this.padW = padW;
        this.inC = inC; this.inH = inH; this.inW = inW;
    }

    @Override
    public Tensor forward(Tensor x) {
        int batch = x.shape[0];
        int inSize = inC * inH * inW;
        int outH = (inH + 2 * padH - kernelH) / strideH + 1;
        int outW = (inW + 2 * padW - kernelW) / strideW + 1;
        int outSize = inC * outH * outW;
        Tensor out = new Tensor(batch, inC, outH, outW);

        if (x.isGPU()) {
            out.toGPU();
            CUDAOps.poolingForward(jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 
                x, out, inC, inH, inW, kernelH, kernelW, outH, outW, padH, padW, strideH, strideW);
            
            if (Torch.is_grad_enabled() && x.requires_grad) {
                out.requires_grad = true;
                out.grad_fn = new Tensor.GradFn(x, out) {
                    public void apply(Tensor outGrad) {
                        Tensor gx = new Tensor(x.shape).toGPU();
                        CUDAOps.poolingBackward(jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 
                            x, out, outGrad, gx, 
                            inC, inH, inW, kernelH, kernelW, outH, outW, padH, padW, strideH, strideW);
                        x.backwardStep(gx);
                    }
                };
            }
            return out;
        }

        x.toCPU();
        int[] counts = new int[batch * outSize];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < inC; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sumv = 0f;
                        int cnt = 0;
                        for (int kh = 0; kh < kernelH; kh++) {
                            for (int kw = 0; kw < kernelW; kw++) {
                                int ih = oh * strideH - padH + kh;
                                int iw2 = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inH && iw2 >= 0 && iw2 < inW) {
                                    sumv += x.data[b * inSize + (c * inH * inW + ih * inW + iw2)];
                                    cnt++;
                                }
                            }
                        }
                        int outIdx = b * outSize + (c * outH * outW + oh * outW + ow);
                        out.data[outIdx] = cnt > 0 ? sumv / cnt : 0f;
                        counts[outIdx] = cnt;
                    }
                }
            }
        }
        if (Torch.is_grad_enabled() && x.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x) {
                public void apply(Tensor outGrad) {
                    Tensor gx = new Tensor(x.shape);
                    for (int b = 0; b < batch; b++) {
                        for (int c = 0; c < inC; c++) {
                            for (int oh = 0; oh < outH; oh++) {
                                for (int ow = 0; ow < outW; ow++) {
                                    int outIdx = b * outSize + (c * outH * outW + oh * outW + ow);
                                    int cnt = counts[outIdx];
                                    if (cnt == 0) continue;
                                    float grad = outGrad.data[outIdx] / cnt;
                                    for (int kh = 0; kh < kernelH; kh++) {
                                        for (int kw = 0; kw < kernelW; kw++) {
                                            int ih = oh * strideH - padH + kh;
                                            int iw2 = ow * strideW - padW + kw;
                                            if (ih >= 0 && ih < inH && iw2 >= 0 && iw2 < inW) {
                                                gx.data[b * inSize + (c * inH * inW + ih * inW + iw2)] += grad;
                                            }
                                        }
                                    }
                                }
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
