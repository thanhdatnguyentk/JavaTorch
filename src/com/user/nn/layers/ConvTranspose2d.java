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
        if (!x.isGPU()) x.toGPU();
        
        int batch = x.shape[0];
        int inC = x.shape[1];
        int inH = x.shape[2];
        int inW = x.shape[3];

        if (inC != inChannels) {
            throw new IllegalArgumentException("Input channels mismatch: expected " + inChannels + " got " + inC);
        }

        int outH = (inH - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outW = (inW - 1) * strideW - 2 * padW + kernelW + outputPadW;

        Tensor out = new Tensor(batch, outChannels, outH, outW);
        out.toGPU();

        Tensor wt = this.weight.getTensor();
        Tensor bt = (this.bias != null) ? this.bias.getTensor() : null;

        // ConvTranspose forward is cuDNN conv backward_data
        CUDAOps.conv2dBackwardData(x, wt, out, 
            outChannels, outH, outW,  // in cuDNN context, out is the 'input' we are reconstructing
            kernelH, kernelW,
            inChannels, inH, inW,     // in cuDNN context, x is the 'dy' gradient
            padH, padW, strideH, strideW);

        if (bt != null) {
            // Apply bias on GPU
            // cuDNN's cudnnAddTensor expects bias of size [1, outC, 1, 1]
            CUDAOps.add(out, bt.reshape(1, outChannels, 1, 1), out);
        }

        if (Torch.is_grad_enabled() && (x.requires_grad || wt.requires_grad || (bt != null && bt.requires_grad))) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x, wt, bt) {
                @Override
                public void apply(Tensor gradOutput) {
                    if (!gradOutput.isGPU()) gradOutput.toGPU();

                    if (x.requires_grad) {
                        Tensor dx = new Tensor(x.shape);
                        dx.toGPU();
                        // Transpose backward for input is forward convolution
                        CUDAOps.conv2dForward(gradOutput, wt, null, dx,
                            outChannels, outH, outW, kernelH, kernelW,
                            inChannels, inH, inW, padH, padW, strideH, strideW);
                        x.backwardStep(dx);
                    }

                    if (wt.requires_grad) {
                        Tensor dw = new Tensor(wt.shape);
                        dw.toGPU();
                        // Transpose backward for weights is backward_filter(dy=input, x=grad_output)
                        // Wait, in regular conv: dw = backward_filter(x=in, dy=grad_out)
                        // In transpose: dw = backward_filter(x=grad_out, dy=in)
                        CUDAOps.conv2dBackwardFilter(gradOutput, x, dw,
                            outChannels, outH, outW, kernelH, kernelW,
                            inChannels, inH, inW, padH, padW, strideH, strideW);
                        wt.backwardStep(dw);
                    }

                    if (bt != null && bt.requires_grad) {
                        Tensor db = new Tensor(bt.shape);
                        db.toGPU();
                        CUDAOps.conv2dBackwardBias(gradOutput, db, outChannels, outH, outW);
                        bt.backwardStep(db);
                    }
                }
            };
        }

        return out;
    }
}
