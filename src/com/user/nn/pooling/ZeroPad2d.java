package com.user.nn.pooling;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class ZeroPad2d extends Module {
    public int padH, padW, inC, inH, inW;

    public ZeroPad2d(int padH, int padW, int inC, int inH, int inW) {
        this.padH = padH; this.padW = padW;
        this.inC = inC; this.inH = inH; this.inW = inW;
    }

    @Override
    public Tensor forward(Tensor x) {
        x.toCPU();
        int batch = x.shape[0];
        int inSize = inC * inH * inW;
        int outH = inH + 2 * padH;
        int outW = inW + 2 * padW;
        int outSize = inC * outH * outW;
        Tensor out = new Tensor(batch, inC, outH, outW);
        if (x.isGPU()) out.toGPU();
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < inC; c++) {
                for (int h = 0; h < inH; h++) {
                    for (int w = 0; w < inW; w++) {
                        float v = x.data[b * inSize + (c * inH * inW + h * inW + w)];
                        int outIdx = b * outSize + (c * outH * outW + (h + padH) * outW + (w + padW));
                        out.data[outIdx] = v;
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
                            for (int h = 0; h < inH; h++) {
                                for (int w = 0; w < inW; w++) {
                                    int outIdx = b * outSize + (c * outH * outW + (h + padH) * outW + (w + padW));
                                    gx.data[b * inSize + (c * inH * inW + h * inW + w)] += outGrad.data[outIdx];
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
