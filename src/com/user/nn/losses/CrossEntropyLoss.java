package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class CrossEntropyLoss extends Module {
    public Tensor forward(Tensor x, int[] targets) {
        return Functional.cross_entropy_tensor(x, targets);
    }
}
