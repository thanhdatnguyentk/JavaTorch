package com.user.nn.metrics;

import com.user.nn.core.Tensor;

/**
 * Classification accuracy metric.
 */
public class Accuracy implements Metric {
    private int correct = 0;
    private int total = 0;

    @Override
    public void update(Tensor preds, Tensor targets) {
        // Assume targets are shaped [N, 1] or [N] containing integer labels
        // Assume preds are shaped [N, C] containing logits
        int bs = preds.shape[0];
        int numClasses = preds.data.length / bs;

        for (int i = 0; i < bs; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int predClass = 0;
            for (int j = 0; j < numClasses; j++) {
                float v = preds.data[i * numClasses + j];
                if (v > maxVal) {
                    maxVal = v;
                    predClass = j;
                }
            }

            int targetClass = (int) targets.data[i];
            if (predClass == targetClass) {
                correct++;
            }
            total++;
        }
    }

    /**
     * Special update for cases where targets are provided as an int array.
     */
    public void update(Tensor preds, int[] targets) {
        int bs = preds.shape[0];
        int numClasses = preds.data.length / bs;

        for (int i = 0; i < bs; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int predClass = 0;
            for (int j = 0; j < numClasses; j++) {
                float v = preds.data[i * numClasses + j];
                if (v > maxVal) {
                    maxVal = v;
                    predClass = j;
                }
            }

            if (predClass == targets[i]) {
                correct++;
            }
            total++;
        }
    }

    @Override
    public float compute() {
        return total == 0 ? 0f : (float) correct / total;
    }

    @Override
    public void reset() {
        correct = 0;
        total = 0;
    }
}
