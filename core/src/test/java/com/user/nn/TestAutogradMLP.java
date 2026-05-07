package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.containers.*;
import com.user.nn.activations.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradMLP {

    @Test
    void testMLPForwardBackward() {
        Torch.manual_seed(42);

        // Layer 1: 2 -> 4
        Linear l1 = new Linear(2, 4, true);
        // Layer 2: 4 -> 2
        Linear l2 = new Linear(4, 2, true);

        Sequential model = new Sequential();
        model.add(l1);
        model.add(new ReLU());
        model.add(l2);

        // Forward pass test
        Tensor x = Torch.tensor(new float[] { 1.5f, -0.5f }, 1, 2);
        Tensor out = model.forward(x);

        // Expected output shape [1, 2]
        assertEquals(2, out.dim(), "MLP output rank");
        assertEquals(1, out.shape[0], "MLP batch size");
        assertEquals(2, out.shape[1], "MLP out channels");

        int[] targets = new int[] { 1 };
        Tensor loss = Functional.cross_entropy_tensor(out, targets);

        assertEquals(1, loss.dim(), "Loss rank");
        assertEquals(1, loss.shape[0], "Loss batch size");

        // Backward pass
        loss.backward();

        // Check gradients populated on weights and biases
        assertNotNull(l1.weight.getTensor().grad, "L1 weight grad should not be null");
        assertNotNull(l1.bias.getTensor().grad, "L1 bias grad should not be null");
        assertNotNull(l2.weight.getTensor().grad, "L2 weight grad should not be null");
        assertNotNull(l2.bias.getTensor().grad, "L2 bias grad should not be null");
    }
}
