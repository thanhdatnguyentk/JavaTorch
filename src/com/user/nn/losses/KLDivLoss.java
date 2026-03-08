package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class KLDivLoss extends Module {
    public Tensor forward(Tensor x, Tensor target) {
        return Functional.kl_div(x, target);
    }
}
