package com.user.nn.models.cv;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.containers.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import com.user.nn.norm.*;
import com.user.nn.pooling.*;
import java.util.*;

/**
 * VGG architecture implementation.
 * Supports VGG11, VGG13, VGG16, VGG19 configurations.
 * Optimized for CIFAR-10 (32x32 input) or generic input sizes.
 */
public class VGG extends Module {
    public Sequential features;
    public Sequential classifier;

    private static final Map<String, Object[]> configs = new HashMap<>();
    static {
        configs.put("VGG11", new Object[]{64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"});
        configs.put("VGG13", new Object[]{64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"});
        configs.put("VGG16", new Object[]{64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"});
        configs.put("VGG19", new Object[]{64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"});
    }

    public VGG(String modelName, int inChannels, int numClasses, boolean batchNorm, int inH, int inW) {
        Object[] cfg = configs.get(modelName);
        if (cfg == null) throw new IllegalArgumentException("Unknown VGG model: " + modelName);

        this.features = new Sequential();
        int currentC = inChannels;
        int currentH = inH;
        int currentW = inW;

        for (Object x : cfg) {
            if (x instanceof String && x.equals("M")) {
                features.add(new MaxPool2d(2, 2, 2, 2, 0, 0, currentC, currentH, currentW));
                currentH /= 2;
                currentW /= 2;
            } else {
                int outC = (int) x;
                features.add(new Conv2d(currentC, outC, 3, 3, currentH, currentW, 1, 1, true));
                if (batchNorm) {
                    features.add(new BatchNorm2d(outC));
                }
                features.add(new ReLU());
                currentC = outC;
            }
        }
        addModule("features", features);

        // For CIFAR-10 (32x32), after 5 max pools (2^5=32), spatial size is 1x1.
        // For ImageNet (224x224), after 5 max pools, spatial size is 7x7.
        int flattenSize = currentC * currentH * currentW;

        this.classifier = new Sequential();
        this.classifier.add(new Linear(flattenSize, 512, true));
        this.classifier.add(new ReLU());
        this.classifier.add(new Dropout(0.5f));
        this.classifier.add(new Linear(512, 512, true));
        this.classifier.add(new ReLU());
        this.classifier.add(new Dropout(0.5f));
        this.classifier.add(new Linear(512, numClasses, true));
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
