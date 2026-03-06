package com.user.nn.metrics;

import com.user.nn.core.Tensor;

/**
 * Common interface for all metrics.
 */
public interface Metric {
    /**
     * Update the internal state with a new batch of predictions and targets.
     * @param preds Model output logits or predictions.
     * @param targets Ground truth values.
     */
    void update(Tensor preds, Tensor targets);

    /**
     * Compute the final metric value based on all updates since the last reset.
     * @return The aggregated metric value.
     */
    float compute();

    /**
     * Reset the internal state (usually called at the start of an epoch).
     */
    void reset();
}
