package com.user.nn.activations;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class LogSoftmax extends Module {
    public int dim;

    public LogSoftmax(int dim) {
        this.dim = dim;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.log_softmax(x, dim);
    }
}
