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
        if (!x.isGPU()) x.toGPU();
        
        int batch = x.shape[0];
        int inC = x.shape[1];
        int inH = x.shape[2];
        int inW = x.shape[3];

        if (inC != inChannels) {
            throw new IllegalArgumentException("Input channels mismatch: expected " + inChannels + " got " + inC);
        }

        int outH = (inH + 2 * padH - kernelH) / strideH + 1;
        int outW = (inW + 2 * padW - kernelW) / strideW + 1;

        Tensor out = new Tensor(batch, outChannels, outH, outW);
        out.toGPU();

        Tensor wt = this.weight.getTensor();
        Tensor bt = (this.bias != null) ? this.bias.getTensor() : null;

        CUDAOps.conv2dForward(x, wt, bt, out, 
            inChannels, inH, inW, 
            kernelH, kernelW, 
            outChannels, outH, outW, 
            padH, padW, strideH, strideW);

        if (Torch.is_grad_enabled() && (x.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x, wt, bt) {
                @Override
                public void apply(Tensor gradOutput) {
                    if (!gradOutput.isGPU()) gradOutput.toGPU();

                    if (wt.requires_grad) {
                        Tensor dw = new Tensor(wt.shape);
                        dw.toGPU();
                        CUDAOps.conv2dBackwardFilter(x, gradOutput, dw,
                            inChannels, inH, inW, kernelH, kernelW,
                            outChannels, outH, outW, padH, padW, strideH, strideW);
                        wt.backwardStep(dw);
                    }

                    if (bt != null && bt.requires_grad) {
                        Tensor db = new Tensor(bt.shape);
                        db.toGPU();
                        CUDAOps.conv2dBackwardBias(gradOutput, db, outChannels, outH, outW);
                        bt.backwardStep(db);
                    }

                    if (x.requires_grad) {
                        Tensor dx = new Tensor(x.shape);
                        dx.toGPU();
                        CUDAOps.conv2dBackwardData(gradOutput, wt, dx,
                            inChannels, inH, inW, kernelH, kernelW,
                            outChannels, outH, outW, padH, padW, strideH, strideW);
                        x.backwardStep(dx);
                    }
                }
            };
        }

        return out;
    }
}
