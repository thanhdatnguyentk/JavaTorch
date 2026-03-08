package com.user.nn.models.generative;

import com.user.nn.core.*;

public class VAE extends NN.Module {
    public NN.Sequential encoder;
    public NN.Linear fc_mu;
    public NN.Linear fc_logvar;
    public NN.Sequential decoder;

    public VAE(NN nn, int inputDim, int latentDim) {
        encoder = new NN.Sequential(
            new NN.Linear(nn, inputDim, 512, true),
            new NN.ReLU()
        );
        fc_mu = new NN.Linear(nn, 512, latentDim, true);
        fc_logvar = new NN.Linear(nn, 512, latentDim, true);
        
        decoder = new NN.Sequential(
            new NN.Linear(nn, latentDim, 512, true),
            new NN.ReLU(),
            new NN.Linear(nn, 512, inputDim, true),
            new NN.Sigmoid()
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
