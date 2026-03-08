package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Bilinear extends Module {
    public int d1, d2, outC;
    public Parameter weight;
    public Parameter bias;

    public Bilinear(int d1, int d2, int outC, boolean useBias) {
        this.d1 = d1;
        this.d2 = d2;
        this.outC = outC;
        NN.Mat w = NN.mat_alloc(outC, d1 * d2);
        NN.mat_rand(w, -0.1f, 0.1f);
        this.weight = new Parameter(new Tensor(w.es, outC, d1, d2));
        addParameter("weight", this.weight);
        if (useBias) {
            NN.Mat b = NN.mat_alloc(1, outC);
            NN.mat_fill(b, 0f);
            this.bias = new Parameter(b);
            addParameter("bias", this.bias);
        }
    }

    public Tensor forward(Tensor x1, Tensor x2) {
        return Torch.bilinear(x1, x2, weight.getTensor(), bias != null ? bias.getTensor() : null);
    }
}
