package com.user.nn.models.cv;

import com.user.nn.core.*;
import java.util.*;

/**
 * ResNet architecture implementation.
 * Supports ResNet-18, ResNet-34, etc.
 * Optimized for CIFAR-10 (32x32 input).
 */
public class ResNet extends NN.Module {
    
    public static class BasicBlock extends NN.Module {
        public NN.Sequential layers;
        public NN.Module downsample;
        public int stride;

        public BasicBlock(NN lib, int inChannels, int outChannels, int inH, int inW, int stride) {
            this.stride = stride;
            this.layers = new NN.Sequential();
            
            // First conv
            layers.add(new NN.Conv2d(lib, inChannels, outChannels, 3, 3, inH, inW, stride, 1, false));
            layers.add(new NN.BatchNorm2d(lib, outChannels));
            layers.add(new NN.ReLU());
            
            int midH = (inH + 2 * 1 - 3) / stride + 1;
            int midW = (inW + 2 * 1 - 3) / stride + 1;

            // Second conv
            layers.add(new NN.Conv2d(lib, outChannels, outChannels, 3, 3, midH, midW, 1, 1, false));
            layers.add(new NN.BatchNorm2d(lib, outChannels));
            
            addModule("layers", layers);

            if (stride != 1 || inChannels != outChannels) {
                NN.Sequential ds = new NN.Sequential();
                ds.add(new NN.Conv2d(lib, inChannels, outChannels, 1, 1, inH, inW, stride, 0, false));
                ds.add(new NN.BatchNorm2d(lib, outChannels));
                this.downsample = ds;
                addModule("downsample", downsample);
            } else {
                this.downsample = null;
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor identity = x;
            if (downsample != null) {
                identity = downsample.forward(x);
            }
            Tensor out = layers.forward(x);
            out = Torch.add(out, identity);
            return Torch.relu(out);
        }
    }

    public NN.Sequential initial;
    public NN.Sequential layer1;
    public NN.Sequential layer2;
    public NN.Sequential layer3;
    public NN.Sequential layer4;
    public NN.Linear fc;

    public ResNet(NN lib, int[] numBlocks, int numClasses, int inH, int inW) {
        // Initial layers for CIFAR-10 (small input)
        this.initial = new NN.Sequential();
        initial.add(new NN.Conv2d(lib, 3, 64, 3, 3, inH, inW, 1, 1, false));
        initial.add(new NN.BatchNorm2d(lib, 64));
        initial.add(new NN.ReLU());
        addModule("initial", initial);

        int curH = inH;
        int curW = inW;
        int curC = 64;

        this.layer1 = makeLayer(lib, curC, 64, numBlocks[0], curH, curW, 1);
        addModule("layer1", layer1);
        curC = 64;

        this.layer2 = makeLayer(lib, curC, 128, numBlocks[1], curH, curW, 2);
        addModule("layer2", layer2);
        curC = 128;
        curH /= 2; curW /= 2;

        this.layer3 = makeLayer(lib, curC, 256, numBlocks[2], curH, curW, 2);
        addModule("layer3", layer3);
        curC = 256;
        curH /= 2; curW /= 2;

        this.layer4 = makeLayer(lib, curC, 512, numBlocks[3], curH, curW, 2);
        addModule("layer4", layer4);
        curC = 512;
        curH /= 2; curW /= 2;

        this.fc = new NN.Linear(lib, 512, numClasses, true);
        addModule("fc", fc);
    }

    private NN.Sequential makeLayer(NN lib, int inChannels, int outChannels, int blocks, int inH, int inW, int stride) {
        NN.Sequential seq = new NN.Sequential();
        seq.add(new BasicBlock(lib, inChannels, outChannels, inH, inW, stride));
        
        int curH = (inH + 2 * 1 - 3) / stride + 1;
        int curW = (inW + 2 * 1 - 3) / stride + 1;

        for (int i = 1; i < blocks; i++) {
            seq.add(new BasicBlock(lib, outChannels, outChannels, curH, curW, 1));
        }
        return seq;
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor out = initial.forward(x);
        out = layer1.forward(out);
        out = layer2.forward(out);
        out = layer3.forward(out);
        out = layer4.forward(out);
        
        // Global Average Pooling
        out = NN.F.adaptive_avg_pool2d(out, 1, 1);
        out = out.view(out.shape[0], -1);
        out = fc.forward(out);
        return out;
    }

    public static ResNet resnet18(NN lib, int numClasses, int inH, int inW) {
        return new ResNet(lib, new int[]{2, 2, 2, 2}, numClasses, inH, inW);
    }

    public static ResNet resnet34(NN lib, int numClasses, int inH, int inW) {
        return new ResNet(lib, new int[]{3, 4, 6, 3}, numClasses, inH, inW);
    }
}
