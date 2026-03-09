package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Conv1d extends Module {
    public int inC, outC, kernel, stride, pad;
    public Parameter weight;
    public Parameter bias;

    public Conv1d(int inC, int outC, int kernel, int stride, int pad, boolean useBias) {
        this.inC = inC;
        this.outC = outC;
        this.kernel = kernel;
        this.stride = stride;
        this.pad = pad;
        NN.Mat w = NN.mat_alloc(outC, inC * kernel);
        this.weight = new Parameter(new Tensor(w.es, outC, inC, kernel));
        Torch.nn.init.kaiming_uniform_(this.weight.getTensor());
        addParameter("weight", this.weight);
        if (useBias) {
            NN.Mat b = NN.mat_alloc(1, outC);
            NN.mat_fill(b, 0f);
            this.bias = new Parameter(b);
            addParameter("bias", this.bias);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.conv1d(x, weight.getTensor(), bias != null ? bias.getTensor() : null, stride, pad);
    }
}
