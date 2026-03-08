package com.user.nn.models.generative;

import com.user.nn.core.*;

public class GAN {
    public static class Generator extends NN.Module {
        public NN.Sequential model;
        public Generator(NN nn, int latentDim, int outputDim) {
            model = new NN.Sequential(
                new NN.Linear(nn, latentDim, 256, true),
                new NN.LeakyReLU(0.2f),
                new NN.Linear(nn, 256, 512, true),
                new NN.LeakyReLU(0.2f),
                new NN.Linear(nn, 512, 1024, true),
                new NN.LeakyReLU(0.2f),
                new NN.Linear(nn, 1024, outputDim, true),
                new NN.Tanh()
            );
            addModule("model", model);
        }
        @Override
        public Tensor forward(Tensor z) { return model.forward(z); }
    }

    public static class Discriminator extends NN.Module {
        public NN.Sequential model;
        public Discriminator(NN nn, int inputDim) {
            model = new NN.Sequential(
                new NN.Linear(nn, inputDim, 512, true),
                new NN.LeakyReLU(0.2f),
                new NN.Dropout(0.3f),
                new NN.Linear(nn, 512, 256, true),
                new NN.LeakyReLU(0.2f),
                new NN.Dropout(0.3f),
                new NN.Linear(nn, 256, 1, true),
                new NN.Sigmoid()
            );
            addModule("model", model);
        }
        @Override
        public Tensor forward(Tensor x) { return model.forward(x); }
    }
}
