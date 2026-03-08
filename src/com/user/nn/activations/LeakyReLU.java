package com.user.nn.activations;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class LeakyReLU extends Module {
    private final float negativeSlope;

    public LeakyReLU(float negativeSlope) {
        this.negativeSlope = negativeSlope;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.leaky_relu(x, negativeSlope);
    }
}
