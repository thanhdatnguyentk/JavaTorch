package com.user.nn.metrics;

import com.user.nn.core.*;
import com.user.nn.dataloaders.Data;

public class Evaluator {

    /**
     * Evaluates a model on a given dataset loader using the specified metric.
     * Handles GPU memory scope automatically.
     *
     * @param model  The neural network model
     * @param loader DataLoader providing the test/validation dataset
     * @param metric The metric to compute (e.g., Accuracy)
     * @return The computed metric value
     */
    public static float evaluate(NN.Module model, Data.DataLoader loader, Accuracy metric) {
        model.eval();
        metric.reset();
        
        // Disable autograd tracking during evaluation for speed and memory efficiency
        boolean prevGrad = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false);

        try {
            for (Tensor[] batch : loader) {
                try (MemoryScope scope = new MemoryScope()) {
                    Tensor xBatch = batch[0];
                    Tensor yBatch = batch[1];
                    
                    scope.track(xBatch);
                    scope.track(yBatch);

                    xBatch.toGPU();

                    int bs = xBatch.shape[0];
                    int[] batchLabels = new int[bs];
                    for (int i = 0; i < bs; i++) {
                        batchLabels[i] = (int) yBatch.data[i];
                    }

                    Tensor logits = model.forward(xBatch);
                    metric.update(logits, batchLabels);
                }
            }
        } finally {
            Torch.set_grad_enabled(prevGrad);
            model.train(); // restore model state
        }

        return metric.compute();
    }
}
