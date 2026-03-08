package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class BCELoss extends Module {
    public Tensor forward(Tensor x, Tensor target) {
        return Functional.binary_cross_entropy(x, target);
    }
}
