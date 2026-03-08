package com.user.nn.models.generative;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;

public class VAE extends Module {
    public Sequential encoder;
    public Linear fc_mu;
    public Linear fc_logvar;
    public Sequential decoder;

    public VAE(int inputDim, int latentDim) {
        encoder = new Sequential(
            new Linear(inputDim, 512, true),
            new ReLU()
        );
        fc_mu = new Linear(512, latentDim, true);
        fc_logvar = new Linear(512, latentDim, true);
        
        decoder = new Sequential(
            new Linear(latentDim, 512, true),
            new ReLU(),
            new Linear(512, inputDim, true),
            new Sigmoid()
        );
        
        addModule("encoder", encoder);
        addModule("fc_mu", fc_mu);
        addModule("fc_logvar", fc_logvar);
        addModule("decoder", decoder);
    }

    public Tensor[] encode(Tensor x) {
        Tensor h = encoder.forward(x);
        return new Tensor[]{ fc_mu.forward(h), fc_logvar.forward(h) };
    }

    public static Tensor reparameterize(Tensor mu, Tensor logvar) {
        Tensor std = Torch.exp(Torch.mul(logvar, 0.5f));
        Tensor eps = Torch.randn(mu.shape).to(mu.device);
        return Torch.add(mu, Torch.mul(std, eps));
    }

    public Tensor decode(Tensor z) {
        return decoder.forward(z);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor[] res = encode(x);
        Tensor z = reparameterize(res[0], res[1]);
        return decode(z);
    }
}
