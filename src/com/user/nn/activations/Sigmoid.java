package com.user.nn.activations;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Sigmoid extends Module {
    @Override
    public Tensor forward(Tensor x) {
        return Torch.sigmoid(x);
    }
}
