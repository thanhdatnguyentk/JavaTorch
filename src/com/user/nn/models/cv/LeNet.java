package com.user.nn.models.cv;

import com.user.nn.core.*;

/**
 * Classic LeNet-5 architecture for 28x28 grayscale images (like MNIST).
 * 
 * Input: 1 x 28 x 28
 * Layer 1: Conv2d(6 channels, 5x5, padding=2) -> 28x28
 *          Tanh
 *          MaxPool2d(2x2, stride=2) -> 14x14
 * Layer 2: Conv2d(16 channels, 5x5, padding=0) -> 10x10
 *          Tanh
 *          MaxPool2d(2x2, stride=2) -> 5x5
 * Layer 3: Linear(16 * 5 * 5 -> 120)
 *          Tanh
 * Layer 4: Linear(120 -> 84)
 *          Tanh
 * Output:  Linear(84 -> 10)
 */
public class LeNet extends NN.Sequential {
    public LeNet(NN lib) {
        // C1: Conv2d (in=1, out=6, kH=5, kW=5, inH=28, inW=28, stride=1, pad=2, bias=true)
        add(new NN.Conv2d(lib, 1, 6, 5, 5, 28, 28, 1, 2, true));
        add(new NN.Tanh());
        
        // S2: Pool2d (poolH=2, poolW=2, strH=2, strW=2, padH=0, padW=0, channels=6, inH=28, inW=28)
        add(new NN.MaxPool2d(2, 2, 2, 2, 0, 0, 6, 28, 28));

        // C3: Conv2d (in=6, out=16, kH=5, kW=5, inH=14, inW=14, stride=1, pad=0, bias=true)
        add(new NN.Conv2d(lib, 6, 16, 5, 5, 14, 14, 1, 0, true));
        add(new NN.Tanh());
        
        // S4: Pool2d (poolH=2, poolW=2, strH=2, strW=2, padH=0, padW=0, channels=16, inH=10, inW=10)
        add(new NN.MaxPool2d(2, 2, 2, 2, 0, 0, 16, 10, 10));

        // Flatten features = 16 * 5 * 5 = 400
        int flattenSize = 400;

        // C5: Linear (400 -> 120)
        add(new NN.Linear(lib, flattenSize, 120, true));
        add(new NN.Tanh());

        // F6: Linear (120 -> 84)
        add(new NN.Linear(lib, 120, 84, true));
        add(new NN.Tanh());

        // OUTPUT: Linear (84 -> 10)
        add(new NN.Linear(lib, 84, 10, true));
    }
}
