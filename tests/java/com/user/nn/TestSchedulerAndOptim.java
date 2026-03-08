package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.optim.*;
import java.util.List;

/**
 * Tests for Scheduler (StepLR) and Optimizer edge cases.
 */
public class TestSchedulerAndOptim {
    static int failures = 0;

    public static void main(String[] args) {
        System.out.println("Running TestSchedulerAndOptim...");

        testStepLRDecay();
        testStepLRInvalidStepSize();
        testOptimizerZeroGrad();
        testAdamConvergence();

        if (failures > 0) {
            System.out.println("TestSchedulerAndOptim FAILED (" + failures + " failures).");
            System.exit(1);
        }
        System.out.println("TestSchedulerAndOptim PASSED.");
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            failures++;
        }
    }

    private static void testStepLRDecay() {
        // Create a simple Linear layer to get an optimizer
        Linear layer = new Linear(2, 1, false);
        Optim.SGD sgd = new Optim.SGD(layer.parameters(), 1.0f);

        Scheduler.StepLR scheduler = new Scheduler.StepLR(sgd, 3, 0.1f);

        // Initial LR = 1.0
        check(Math.abs(sgd.getLearningRate() - 1.0f) < 1e-6f, "Initial LR should be 1.0");

        // Steps 1, 2: no decay
        scheduler.step(); // epoch 1
        scheduler.step(); // epoch 2
        check(Math.abs(sgd.getLearningRate() - 1.0f) < 1e-6f, "LR after 2 steps should still be 1.0");

        // Step 3: decay to 0.1
        scheduler.step(); // epoch 3
        check(Math.abs(sgd.getLearningRate() - 0.1f) < 1e-6f, "LR after 3 steps should be 0.1, got " + sgd.getLearningRate());

        // Steps 4, 5: no decay
        scheduler.step(); // epoch 4
        scheduler.step(); // epoch 5
        check(Math.abs(sgd.getLearningRate() - 0.1f) < 1e-6f, "LR after 5 steps should be 0.1");

        // Step 6: decay to 0.01
        scheduler.step(); // epoch 6
        check(Math.abs(sgd.getLearningRate() - 0.01f) < 1e-5f, "LR after 6 steps should be 0.01, got " + sgd.getLearningRate());

        System.out.println("  StepLR decay OK");
    }

    private static void testStepLRInvalidStepSize() {
        Linear layer = new Linear(2, 1, false);
        Optim.SGD sgd = new Optim.SGD(layer.parameters(), 0.1f);

        // stepSize = 0 should throw
        boolean threw = false;
        try {
            new Scheduler.StepLR(sgd, 0, 0.1f);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "StepLR with stepSize=0 should throw IllegalArgumentException");

        // stepSize = -5 should throw
        threw = false;
        try {
            new Scheduler.StepLR(sgd, -5, 0.1f);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "StepLR with stepSize=-5 should throw IllegalArgumentException");

        System.out.println("  StepLR invalid stepSize OK");
    }

    private static void testOptimizerZeroGrad() {
        Linear layer = new Linear(2, 1, true);
        Optim.Adam adam = new Optim.Adam(layer.parameters(), 0.01f);

        // Forward + backward to populate grads
        Tensor x = Torch.tensor(new float[]{1f, 2f}, 1, 2);
        Tensor out = layer.forward(x);
        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        // Check grads exist
        check(layer.weight.getTensor().grad != null, "weight grad should exist after backward");

        // Zero grad should clear them
        adam.zero_grad();
        for (Parameter p : layer.parameters()) {
            check(p.getTensor().grad == null, "grad should be null after zero_grad");
        }

        System.out.println("  Optimizer zero_grad OK");
    }

    private static void testAdamConvergence() {
        // Minimize f(x) = (x - 3)^2 with Adam
        Linear layer = new Linear(1, 1, false);
        Tensor w = layer.weight.getTensor();
        w.data[0] = 10f; // start far from optimum
        w.requires_grad = true;

        Optim.Adam adam = new Optim.Adam(layer.parameters(), 0.1f);

        for (int i = 0; i < 200; i++) {
            adam.zero_grad();
            // loss = (w - 3)^2
            Tensor target = Torch.tensor(new float[]{3f}, 1);
            Tensor diff = Torch.sub(w, target);
            Tensor loss = Torch.sumTensor(Torch.mul(diff, diff));
            loss.backward();
            adam.step();
        }

        check(Math.abs(w.data[0] - 3f) < 0.1f,
              "Adam should converge near 3.0, got " + w.data[0]);

        System.out.println("  Adam convergence OK");
    }
}
