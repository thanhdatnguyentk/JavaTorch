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
        // Read data without mutating tensor device state
        float[] predsData = readData(preds);
        targets.toCPU(); // targets is not part of autograd, safe to mutate
        int bs = preds.shape[0];
        int numClasses = predsData.length / bs;

        for (int i = 0; i < bs; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int predClass = 0;
            for (int j = 0; j < numClasses; j++) {
                float v = predsData[i * numClasses + j];
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
        // Read data without mutating tensor device state
        float[] predsData = readData(preds);
        int bs = preds.shape[0];
        int numClasses = predsData.length / bs;

        for (int i = 0; i < bs; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int predClass = 0;
            for (int j = 0; j < numClasses; j++) {
                float v = predsData[i * numClasses + j];
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

    /**
     * Read tensor data to a float array without mutating device state.
     */
    private static float[] readData(Tensor t) {
        if (t.isGPU()) {
            float[] data = new float[t.data.length];
            com.user.nn.core.CUDAOps.syncComputeStream();
            jcuda.runtime.JCuda.cudaMemcpy(
                jcuda.Pointer.to(data),
                t.getDevicePointer(),
                (long) t.numel() * jcuda.Sizeof.FLOAT,
                jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
            );
            return data;
        }
        return t.data;
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
