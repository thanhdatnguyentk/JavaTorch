package com.user.nn;

import java.util.List;

/**
 * Optimizers for training neural networks.
 * Provides SGD (with optional momentum) and Adam.
 */
public class optim {

    /**
     * Abstract base class for all optimizers.
     */
    public static abstract class Optimizer {
        protected List<nn.Parameter> params;

        public Optimizer(List<nn.Parameter> params) {
            this.params = params;
        }

        /** Zero out gradients of all parameters. */
        public void zero_grad() {
            for (nn.Parameter p : params) {
                Tensor t = p.getTensor();
                t.grad = null;
            }
        }

        /** Perform a single optimization step. */
        public abstract void step();
    }

    /**
     * Stochastic Gradient Descent with optional momentum.
     * update rule:
     * v = momentum * v + grad
     * param -= lr * v
     */
    public static class SGD extends Optimizer {
        private float lr;
        private float momentum;
        private float[][] velocities; // velocity buffers per parameter

        public SGD(List<nn.Parameter> params, float lr) {
            this(params, lr, 0f);
        }

        public SGD(List<nn.Parameter> params, float lr, float momentum) {
            super(params);
            this.lr = lr;
            this.momentum = momentum;
            // Initialize velocity buffers
            this.velocities = new float[params.size()][];
            for (int i = 0; i < params.size(); i++) {
                this.velocities[i] = new float[params.get(i).getTensor().data.length];
            }
        }

        @Override
        public void step() {
            for (int i = 0; i < params.size(); i++) {
                Tensor t = params.get(i).getTensor();
                if (t.grad == null)
                    continue;
                float[] v = velocities[i];
                for (int j = 0; j < t.data.length; j++) {
                    v[j] = momentum * v[j] + t.grad.data[j];
                    t.data[j] -= lr * v[j];
                }
            }
        }
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

        public Adam(List<nn.Parameter> params, float lr) {
            this(params, lr, 0.9f, 0.999f, 1e-8f);
        }

        public Adam(List<nn.Parameter> params, float lr, float beta1, float beta2, float eps) {
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
            for (int i = 0; i < params.size(); i++) {
                Tensor t = params.get(i).getTensor();
                if (t.grad == null)
                    continue;
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
            }
        }
    }
}
