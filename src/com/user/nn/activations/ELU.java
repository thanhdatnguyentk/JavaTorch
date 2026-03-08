package com.user.nn.activations;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class ELU extends Module {
    public float alpha;

    public ELU(float alpha) {
        this.alpha = alpha;
    }

    public ELU() {
        this(1.0f);
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.elu(x, alpha);
    }
}
