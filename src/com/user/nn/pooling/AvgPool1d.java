package com.user.nn.pooling;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class AvgPool1d extends Module {
    public int kernel, stride, pad;

    public AvgPool1d(int kernel, int stride, int pad) {
        this.kernel = kernel; this.stride = stride; this.pad = pad;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.avg_pool1d(x, kernel, stride, pad);
    }
}
