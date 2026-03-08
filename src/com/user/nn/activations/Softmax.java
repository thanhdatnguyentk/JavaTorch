package com.user.nn.activations;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Softmax extends Module {
    public int dim;

    public Softmax(int dim) {
        this.dim = dim;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.softmax(x, dim);
    }
}
