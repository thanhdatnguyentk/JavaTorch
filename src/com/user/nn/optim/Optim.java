package com.user.nn.optim;

import com.user.nn.core.*;
import java.util.*;

/**
 * Optimizers for training neural networks.
 * Provides SGD (with optional momentum) and Adam.
 */
public class Optim {

    /**
     * Abstract base class for all optimizers.
     */
    public static abstract class Optimizer {
        protected List<Parameter> parameters;

        public Optimizer(List<Parameter> parameters) {
            this.parameters = parameters;
        }

        /** Zero out gradients of all parameters. */
        public void zero_grad() {
            for (Parameter p : parameters) {
                Tensor t = p.getTensor();
                t.grad = null;
            }
        }

        /** Perform a single optimization step. */
        public abstract void step();

        public abstract float getLearningRate();
        public abstract void setLearningRate(float lr);
    }

    /**
     * Stochastic Gradient Descent (SGD) with optional momentum.
     */
    public static class SGD extends Optimizer {
        private float lr;
        private final float momentum;

        // Stores velocity for momentum per parameter
        private Map<Parameter, Tensor> v;

        public SGD(List<Parameter> parameters, float lr) {
            this(parameters, lr, 0.0f);
        }

        public SGD(List<Parameter> parameters, float lr, float momentum) {
            super(parameters);
            this.lr = lr;
            this.momentum = momentum;
            if (this.momentum > 0) {
                this.v = new HashMap<>();
                for (Parameter p : parameters) {
                    this.v.put(p, Torch.zeros(p.getTensor().shape));
                }
            }
        }

        @Override
        public void step() {
            for (Parameter p : parameters) {
                Tensor t = p.getTensor();
                if (t.grad == null)
                    continue;

                t.toCPU();
                t.grad.toCPU();

                if (momentum > 0) {
                    Tensor vel = v.get(p);
                    vel.toCPU();
                    for (int i = 0; i < vel.data.length; i++) {
                        vel.data[i] = momentum * vel.data[i] + t.grad.data[i];
                        t.data[i] -= lr * vel.data[i];
                    }
                    vel.markDirtyOnCPU();
                } else {
                    for (int i = 0; i < t.data.length; i++) {
                        t.data[i] -= lr * t.grad.data[i];
                    }
                }

                t.markDirtyOnCPU();
            }
        }

        @Override
        public float getLearningRate() { return lr; }

        @Override
        public void setLearningRate(float lr) { this.lr = lr; }
    }

    /**
     * Adam optimizer (Adaptive Moment Estimation).
     * Kingma & Ba, 2014.
     */
    public static class Adam extends Optimizer {
        private float lr;
        private float beta1;
        private float beta2;
        private float eps;
        private float[][] m; // first moment
        private float[][] v; // second moment
        private int stepCount;

        public Adam(List<Parameter> params, float lr) {
            this(params, lr, 0.9f, 0.999f, 1e-8f);
        }

            // Convenience constructor without eps (uses default eps)
            public Adam(List<Parameter> params, float lr, float beta1, float beta2) {
                this(params, lr, beta1, beta2, 1e-8f);
            }

        public Adam(List<Parameter> params, float lr, float beta1, float beta2, float eps) {
            super(params);
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.stepCount = 0;
            this.m = new float[params.size()][];
            this.v = new float[params.size()][];
            for (int i = 0; i < params.size(); i++) {
                int n = params.get(i).getTensor().data.length;
                this.m[i] = new float[n];
                this.v[i] = new float[n];
            }
        }

        @Override
        public void step() {
            stepCount++;
            float bc1 = 1f - (float) Math.pow(beta1, stepCount); // bias correction
            float bc2 = 1f - (float) Math.pow(beta2, stepCount);
            for (int i = 0; i < parameters.size(); i++) {
                Tensor t = parameters.get(i).getTensor();
                if (t.grad == null)
                    continue;
                
                t.toCPU();
                t.grad.toCPU();

                float[] mi = this.m[i];
                float[] vi = this.v[i];
                for (int j = 0; j < t.data.length; j++) {
                    float g = t.grad.data[j];
                    mi[j] = beta1 * mi[j] + (1f - beta1) * g;
                    vi[j] = beta2 * vi[j] + (1f - beta2) * g * g;
                    float mHat = mi[j] / bc1;
                    float vHat = vi[j] / bc2;
                    t.data[j] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
                }

                t.markDirtyOnCPU();
            }
        }

        @Override
        public float getLearningRate() { return lr; }

        @Override
        public void setLearningRate(float lr) { this.lr = lr; }
    }
}
