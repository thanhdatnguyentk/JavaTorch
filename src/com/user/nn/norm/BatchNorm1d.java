package com.user.nn.norm;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class BatchNorm1d extends Module {
    public int numFeatures;
    public Parameter weight;
    public Parameter bias;
    public float[] runningMean;
    public float[] runningVar;
    public float eps = 1e-5f;
    public float momentum = 0.1f;

    public BatchNorm1d(int numFeatures, boolean affine) {
        this.numFeatures = numFeatures;
        runningMean = new float[numFeatures];
        runningVar = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            runningMean[i] = 0f;
            runningVar[i] = 1f;
        }
        if (affine) {
            NN.Mat gw = NN.mat_alloc(1, numFeatures);
            NN.mat_fill(gw, 1.0f);
            weight = new Parameter(gw);
            addParameter("weight", weight);
            NN.Mat gb = NN.mat_alloc(1, numFeatures);
            NN.mat_fill(gb, 0.0f);
            bias = new Parameter(gb);
            addParameter("bias", bias);
        }
    }

    @Override
    public NN.Mat forward(NN.Mat x) {
        if (x.cols != numFeatures)
            throw new IllegalArgumentException("BatchNorm1d: feature mismatch");
        int batch = x.rows;
        NN.Mat out = new NN.Mat();
        out.rows = batch;
        out.cols = numFeatures;
        out.es = new float[batch * numFeatures];

        float[] useMean;
        float[] useVar;

        if (training) {
            float[] mean = new float[numFeatures];
            float[] var = new float[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                float s = 0f;
                for (int i = 0; i < batch; i++)
                    s += x.es[i * numFeatures + j];
                mean[j] = s / batch;
            }
            for (int j = 0; j < numFeatures; j++) {
                float s = 0f;
                for (int i = 0; i < batch; i++) {
                    float d = x.es[i * numFeatures + j] - mean[j];
                    s += d * d;
                }
                var[j] = s / batch;
            }
            for (int j = 0; j < numFeatures; j++) {
                runningMean[j] = momentum * mean[j] + (1 - momentum) * runningMean[j];
                runningVar[j] = (momentum * var[j]) + (1 - momentum) * runningVar[j];
            }
            useMean = mean;
            useVar = var;
        } else {
            useMean = runningMean;
            useVar = runningVar;
        }

        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < numFeatures; j++) {
                float val = (x.es[i * numFeatures + j] - useMean[j]) / (float) Math.sqrt(useVar[j] + eps);
                if (weight != null)
                    val = val * weight.data.es[j] + bias.data.es[j];
                out.es[i * numFeatures + j] = val;
            }
        }
        return out;
    }
}
