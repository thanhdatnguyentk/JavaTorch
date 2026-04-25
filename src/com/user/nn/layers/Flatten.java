package com.user.nn.layers;

import com.user.nn.core.Module;
import com.user.nn.core.Tensor;

public class Flatten extends Module {
    @Override
    public Tensor forward(Tensor input) {
        int batchSize = input.shape[0];
        int flatDim = input.numel() / batchSize;
        return input.reshape(batchSize, flatDim);
    }
}
