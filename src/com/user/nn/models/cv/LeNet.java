package com.user.nn.models.cv;

import com.user.nn.core.*;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import com.user.nn.pooling.*;

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
public class LeNet extends Sequential {
    public LeNet() {
        // C1: Conv2d (inChannels=1, outChannels=32, kernelH=5, kernelW=5, strideH=1, strideW=1, padH=2, padW=2, biasFlag=true)
        add(new Conv2d(1, 32, 5, 5, 1, 1, 2, 2, true));
        add(new ReLU());
        
        // S2: Pool2d (kernelH=2, kernelW=2, strideH=2, strideW=2, padH=0, padW=0, inC=32, inH=28, inW=28)
        add(new MaxPool2d(2, 2, 2, 2, 0, 0, 32, 28, 28));

        // C3: Conv2d (inChannels=32, outChannels=64, kernelH=5, kernelW=5, strideH=1, strideW=1, padH=0, padW=0, biasFlag=true)
        add(new Conv2d(32, 64, 5, 5, 1, 1, 0, 0, true));
        add(new ReLU());
        
        // S4: Pool2d (kernelH=2, kernelW=2, strideH=2, strideW=2, padH=0, padW=0, inC=64, inH=10, inW=10)
        add(new MaxPool2d(2, 2, 2, 2, 0, 0, 64, 10, 10));

        // Flatten features = 64 * 5 * 5 = 1600
        int flattenSize = 1600;

        // Flatten to (batch, 1600)
        add(new com.user.nn.containers.Flatten());

        // C5: Linear (1600 -> 256)
        add(new Linear(flattenSize, 256, true));
        add(new ReLU());
        add(new Dropout(0.2f));

        // F6: Linear (256 -> 128)
        add(new Linear(256, 128, true));
        add(new ReLU());
        add(new Dropout(0.2f));

        // OUTPUT: Linear (128 -> 10)
        add(new Linear(128, 10, true));
    }
}
