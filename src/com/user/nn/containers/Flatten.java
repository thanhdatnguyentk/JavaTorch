package com.user.nn.containers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Flatten extends Module {
    @Override
    public Tensor forward(Tensor x) {
        if (x.shape.length <= 2) return x;
        int batch = x.shape[0];
        int prod = 1;
        for (int i = 1; i < x.shape.length; i++) prod *= x.shape[i];
        return Torch.reshape(x, batch, prod);
    }
}
