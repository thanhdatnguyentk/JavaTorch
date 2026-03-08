package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class L1Loss extends Module {
    public Tensor forward(Tensor x, Tensor target) {
        return Functional.l1_loss(x, target);
    }
}
