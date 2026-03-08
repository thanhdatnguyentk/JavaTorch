package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class BCEWithLogitsLoss extends Module {
    public Tensor forward(Tensor x, Tensor target) {
        return Functional.binary_cross_entropy_with_logits(x, target);
    }
}
