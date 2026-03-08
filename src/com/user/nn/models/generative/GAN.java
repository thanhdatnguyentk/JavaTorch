package com.user.nn.models.generative;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;

public class GAN {
    public static class Generator extends Module {
        public Sequential model;
        public Generator(int latentDim, int outputDim) {
            model = new Sequential(
                new Linear(latentDim, 256, true),
                new LeakyReLU(0.2f),
                new Linear(256, 512, true),
                new LeakyReLU(0.2f),
                new Linear(512, 1024, true),
                new LeakyReLU(0.2f),
                new Linear(1024, outputDim, true),
                new Tanh()
            );
            addModule("model", model);
        }
        @Override
        public Tensor forward(Tensor z) { return model.forward(z); }
    }

    public static class Discriminator extends Module {
        public Sequential model;
        public Discriminator(int inputDim) {
            model = new Sequential(
                new Linear(inputDim, 512, true),
                new LeakyReLU(0.2f),
                new Dropout(0.3f),
                new Linear(512, 256, true),
                new LeakyReLU(0.2f),
                new Dropout(0.3f),
                new Linear(256, 1, true),
                new Sigmoid()
            );
            addModule("model", model);
        }
        @Override
        public Tensor forward(Tensor x) { return model.forward(x); }
    }
}
