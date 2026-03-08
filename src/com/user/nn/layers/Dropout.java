package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Dropout extends Module {
    public float p;

    public Dropout(float p) {
        this.p = p;
    }

    public Dropout(float p, long seed) {
        this.p = p;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.dropout(x, p, training);
    }
}
