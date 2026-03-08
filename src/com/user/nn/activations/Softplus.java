package com.user.nn.activations;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Softplus extends Module {
    @Override
    public Tensor forward(Tensor x) {
        x.toCPU();
        Tensor out = new Tensor(x.shape);
        for (int i = 0; i < x.data.length; i++) {
            out.data[i] = (float) Math.log(1.0 + Math.exp(x.data[i]));
        }
        if (x.isGPU()) out.toGPU();
        if (Torch.is_grad_enabled() && x.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(x) {
                public void apply(Tensor outGrad) {
                    Tensor gx = new Tensor(x.shape);
                    for (int i = 0; i < gx.data.length; i++) {
                        float expX = (float) Math.exp(x.data[i]);
                        gx.data[i] = (expX / (1.0f + expX)) * outGrad.data[i];
                    }
                    x.backwardStep(gx);
                }
            };
        }
        return out;
    }
}
