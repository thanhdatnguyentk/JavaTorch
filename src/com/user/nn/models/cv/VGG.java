package com.user.nn.models.cv;

import com.user.nn.core.*;
import java.util.*;

/**
 * VGG architecture implementation.
 * Supports VGG11, VGG13, VGG16, VGG19 configurations.
 * Optimized for CIFAR-10 (32x32 input) or generic input sizes.
 */
public class VGG extends NN.Module {
    public NN.Sequential features;
    public NN.Sequential classifier;

    private static final Map<String, Object[]> configs = new HashMap<>();
    static {
        configs.put("VGG11", new Object[]{64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"});
        configs.put("VGG13", new Object[]{64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"});
        configs.put("VGG16", new Object[]{64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"});
        configs.put("VGG19", new Object[]{64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"});
    }

    public VGG(NN lib, String modelName, int inChannels, int numClasses, boolean batchNorm, int inH, int inW) {
        Object[] cfg = configs.get(modelName);
        if (cfg == null) throw new IllegalArgumentException("Unknown VGG model: " + modelName);

        this.features = new NN.Sequential();
        int currentC = inChannels;
        int currentH = inH;
        int currentW = inW;

        for (Object x : cfg) {
            if (x instanceof String && x.equals("M")) {
                features.add(new NN.MaxPool2d(2, 2, 2, 2, 0, 0, currentC, currentH, currentW));
                currentH /= 2;
                currentW /= 2;
            } else {
                int outC = (int) x;
                features.add(new NN.Conv2d(lib, currentC, outC, 3, 3, currentH, currentW, 1, 1, true));
                if (batchNorm) {
                    features.add(new NN.BatchNorm2d(lib, outC));
                }
                features.add(new NN.ReLU());
                currentC = outC;
            }
        }
        addModule("features", features);

        // For CIFAR-10 (32x32), after 5 max pools (2^5=32), spatial size is 1x1.
        // For ImageNet (224x224), after 5 max pools, spatial size is 7x7.
        int flattenSize = currentC * currentH * currentW;

        this.classifier = new NN.Sequential();
        this.classifier.add(new NN.Linear(lib, flattenSize, 512, true));
        this.classifier.add(new NN.ReLU());
        this.classifier.add(new NN.Dropout(0.5f));
        this.classifier.add(new NN.Linear(lib, 512, 512, true));
        this.classifier.add(new NN.ReLU());
        this.classifier.add(new NN.Dropout(0.5f));
        this.classifier.add(new NN.Linear(lib, 512, numClasses, true));
        addModule("classifier", classifier);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor out = features.forward(x);
        int b = out.shape[0];
        out = out.view(b, -1);
        out = classifier.forward(out);
        return out;
    }
}
