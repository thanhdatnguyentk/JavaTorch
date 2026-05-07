package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.Linear;
import com.user.nn.optim.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestSchedulerAndOptim {

    @Test
    void testStepLRDecay() {
        Linear layer = new Linear(2, 1, false);
        Optim.SGD sgd = new Optim.SGD(layer.parameters(), 1.0f);
        Scheduler.StepLR scheduler = new Scheduler.StepLR(sgd, 3, 0.1f);

        assertEquals(1.0f, sgd.getLearningRate(), 1e-6f, "Initial LR should be 1.0");

        scheduler.step(); // epoch 1
        scheduler.step(); // epoch 2
        assertEquals(1.0f, sgd.getLearningRate(), 1e-6f, "LR after 2 steps should still be 1.0");

        scheduler.step(); // epoch 3
        assertEquals(0.1f, sgd.getLearningRate(), 1e-6f, "LR after 3 steps should be 0.1");

        scheduler.step(); // epoch 4
        scheduler.step(); // epoch 5
        assertEquals(0.1f, sgd.getLearningRate(), 1e-6f, "LR after 5 steps should be 0.1");

        scheduler.step(); // epoch 6
        assertEquals(0.01f, sgd.getLearningRate(), 1e-5f, "LR after 6 steps should be 0.01");
    }

    @Test
    void testStepLRInvalidStepSize() {
        Linear layer = new Linear(2, 1, false);
        Optim.SGD sgd = new Optim.SGD(layer.parameters(), 0.1f);

        assertThrows(IllegalArgumentException.class, () -> new Scheduler.StepLR(sgd, 0, 0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Scheduler.StepLR(sgd, -5, 0.1f));
    }

    @Test
    void testOptimizerZeroGrad() {
        Linear layer = new Linear(2, 1, true);
        Optim.Adam adam = new Optim.Adam(layer.parameters(), 0.01f);

        Tensor x = Torch.tensor(new float[]{1f, 2f}, 1, 2);
        Tensor out = layer.forward(x);
        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(layer.weight.getTensor().grad, "weight grad should exist after backward");

        adam.zero_grad();
        for (Parameter p : layer.parameters()) {
            assertNull(p.getTensor().grad, "grad should be null after zero_grad");
        }
    }

    @Test
    void testAdamConvergence() {
        // Minimize f(x) = (x - 3)^2 with Adam
        Linear layer = new Linear(1, 1, false);
        Tensor w = layer.weight.getTensor();
        w.data[0] = 10f; // start far from optimum
        w.requires_grad = true;

        Optim.Adam adam = new Optim.Adam(layer.parameters(), 0.1f);

        for (int i = 0; i < 150; i++) {
            adam.zero_grad();
            // loss = (w - 3)^2
            Tensor target = Torch.tensor(new float[]{3f}, 1);
            Tensor diff = Torch.sub(w, target);
            Tensor loss = Torch.sumTensor(Torch.mul(diff, diff));
            loss.backward();
            adam.step();
        }

        assertEquals(3.0f, w.data[0], 0.1f, "Adam should converge near 3.0");
    }
}
