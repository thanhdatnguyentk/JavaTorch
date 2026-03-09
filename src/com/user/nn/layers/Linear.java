package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Linear extends Module {
    public int inFeatures;
    public int outFeatures;
    public Parameter weight;
    public Parameter bias;

    public Linear(int inFeatures, int outFeatures, boolean useBias) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        NN.Mat w = NN.mat_alloc(inFeatures, outFeatures);
        this.weight = new Parameter(w);
        Torch.nn.init.kaiming_uniform_(this.weight.getTensor());
        addParameter("weight", this.weight);
        if (useBias) {
            NN.Mat b = NN.mat_alloc(1, outFeatures);
            NN.mat_fill(b, 0.0f);
            this.bias = new Parameter(b);
            addParameter("bias", this.bias);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.shape[input.shape.length - 1] != inFeatures) {
            throw new IllegalArgumentException("Input features mismatch: expected " + inFeatures + " got "
                    + input.shape[input.shape.length - 1]);
        }
        Tensor w = this.weight.getTensor();
        Tensor out = Torch.matmul(input, w);
        if (this.bias != null) {
            Tensor b = this.bias.getTensor();
            out = Torch.add(out, b);
        }
        return out;
    }
}
