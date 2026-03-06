package com.user.nn.metrics;

import com.user.nn.core.Tensor;

/**
 * Mean Squared Error metric for regression.
 */
public class MeanSquaredError implements Metric {
    private float sumSquaredError = 0f;
    private int totalSamples = 0;

    @Override
    public void update(Tensor preds, Tensor targets) {
        preds.toCPU();
        targets.toCPU();
        if (preds.numel() != targets.numel()) {
            throw new IllegalArgumentException("Predictions and targets must have same total elements.");
        }
        for (int i = 0; i < preds.data.length; i++) {
            float diff = preds.data[i] - targets.data[i];
            sumSquaredError += diff * diff;
        }
        totalSamples += preds.data.length;
    }

    @Override
    public float compute() {
        return totalSamples == 0 ? 0f : sumSquaredError / totalSamples;
    }

    @Override
    public void reset() {
        sumSquaredError = 0f;
        totalSamples = 0;
    }
}
