package com.user.nn.optim;

/**
 * Learning rate schedulers for optimizers.
 */
public abstract class Scheduler {
    protected Optim.Optimizer optimizer;

    public Scheduler(Optim.Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public abstract void step();

    /**
     * Decays the learning rate of each parameter group by gamma every step_size epochs.
     */
    public static class StepLR extends Scheduler {
        private int stepSize;
        private float gamma;
        private int lastEpoch = 0;

        public StepLR(Optim.Optimizer optimizer, int stepSize, float gamma) {
            super(optimizer);
            if (stepSize <= 0) throw new IllegalArgumentException("stepSize must be > 0, got: " + stepSize);
            this.stepSize = stepSize;
            this.gamma = gamma;
        }

        @Override
        public void step() {
            lastEpoch++;
            if (lastEpoch % stepSize == 0) {
                float currentLr = optimizer.getLearningRate();
                optimizer.setLearningRate(currentLr * gamma);
            }
        }
    }
}
