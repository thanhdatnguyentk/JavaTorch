package com.user.nn.pooling;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class AdaptiveAvgPool2d extends Module {
    public int outputH, outputW;

    public AdaptiveAvgPool2d(int outputH, int outputW) {
        this.outputH = outputH; this.outputW = outputW;
    }

    @Override
    public Tensor forward(Tensor x) {
        return Torch.adaptive_avg_pool2d(x, new int[] { outputH, outputW });
    }
}
