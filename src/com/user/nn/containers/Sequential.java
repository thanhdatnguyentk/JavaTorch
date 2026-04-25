package com.user.nn.containers;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.Conv2d;
import com.user.nn.activations.ReLU;
import java.util.ArrayList;
import java.util.List;

public class Sequential extends Module {
    private final List<Module> list = new ArrayList<>();

    public Sequential() {
    }

    public Sequential(Module... modules) {
        for (Module m : modules) {
            add(m);
        }
    }

    public void add(Module m) {
        String name = "" + list.size();
        list.add(m);
        addModule(name, m);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor out = x;
        for (int i = 0; i < list.size(); i++) {
            Module m = list.get(i);
            
            // Kernel Fusion: detect Conv2d(bias=true) + ReLU on GPU
            if (out.isGPU() && m instanceof Conv2d && i + 1 < list.size() && list.get(i + 1) instanceof ReLU) {
                Conv2d conv = (Conv2d) m;
                if (conv.bias != null) {
                    int batch = out.shape[0];
                    int inH = out.shape[2];
                    int inW = out.shape[3];
                    int outH = (inH + 2 * conv.padH - conv.kernelH) / conv.strideH + 1;
                    int outW = (inW + 2 * conv.padW - conv.kernelW) / conv.strideW + 1;
                    int ksz = conv.inChannels * conv.kernelH * conv.kernelW;
                    int outSize = conv.outChannels * outH * outW;

                    Tensor wt = conv.weight.getTensor();
                    Tensor bt = conv.bias.getTensor();
                    Tensor fusedOut = new Tensor(batch, outSize);
                    fusedOut.toGPU();

                    Tensor wtT = new Tensor(new int[]{conv.outChannels, ksz});
                    wtT.toGPU();
                    CUDAOps.transpose(wt, wtT);
                    
                    CUDAOps.conv2dBiasReluForward(out, wtT, bt, fusedOut,
                        conv.inChannels, inH, inW,
                        conv.kernelH, conv.kernelW,
                        conv.outChannels, outH, outW,
                        conv.padH, conv.padW, conv.strideH, conv.strideW);
                    
                    wtT.close();
                    
                    if (Torch.is_grad_enabled() && (out.requires_grad || wt.requires_grad || bt.requires_grad)) {
                        fusedOut.requires_grad = true;
                        final Tensor convInput = out;
                        fusedOut.grad_fn = new Tensor.GradFn(convInput, wt, bt) {
                            public void apply(Tensor outGrad) {
                                fusedOut.toCPU();
                                outGrad.toCPU();
                                for (int j = 0; j < fusedOut.data.length; j++) {
                                    if (fusedOut.data[j] <= 0) outGrad.data[j] = 0;
                                }
                                if (convInput.isGPU()) outGrad.toGPU();
                                conv.forward(convInput).grad_fn.apply(outGrad);
                            }
                        };
                    }
                    
                    out = fusedOut;
                    i++;
                    continue;
                }
            }
            
            out = m.forward(out);
        }
        return out;
    }

    @Override
    public NN.Mat forward(NN.Mat x) {
        NN.Mat out = x;
        for (Module m : list) {
            out = m.forward(out);
        }
        return out;
    }
}
