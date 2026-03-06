package com.user.nn.metrics;

import com.user.nn.core.Tensor;

/**
 * Mean Absolute Error metric for regression.
 */
public class MeanAbsoluteError implements Metric {
    private float sumAbsoluteError = 0f;
    private int totalSamples = 0;

    @Override
    public void update(Tensor preds, Tensor targets) {
        if (preds.data.length != targets.data.length) {
            throw new IllegalArgumentException("Predictions and targets must have same total elements.");
        }
        for (int i = 0; i < preds.data.length; i++) {
            sumAbsoluteError += Math.abs(preds.data[i] - targets.data[i]);
        }
        totalSamples += preds.data.length;
    }

    @Override
    public float compute() {
        return totalSamples == 0 ? 0f : sumAbsoluteError / totalSamples;
    }

    @Override
    public void reset() {
        sumAbsoluteError = 0f;
        totalSamples = 0;
    }
}
