package com.user.nn.pooling;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class MaxPool1d extends Module {
    public int kernel, stride, pad;

    public MaxPool1d(int kernel, int stride, int pad) {
        this.kernel = kernel; this.stride = stride; this.pad = pad;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.max_pool1d(x, kernel, stride, pad);
    }
}
