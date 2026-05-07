package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradActivations {

    @Test
    void testRelu() {
        Tensor a = Torch.tensor(new float[] { -2, -1, 0, 1, 2, 3 }, 2, 3);
        a.requires_grad = true;
        Tensor b = Torch.relu(a);
        Tensor loss = Torch.sumTensor(b);
        loss.backward();

        assertNotNull(a.grad, "grad present");
        // derivative of ReLU is 0 if x <= 0, else 1
        assertEquals(0f, a.grad.data[0], 1e-6f, "relu grad neg");
        assertEquals(0f, a.grad.data[1], 1e-6f, "relu grad neg");
        assertEquals(0f, a.grad.data[2], 1e-6f, "relu grad zero");
        assertEquals(1f, a.grad.data[3], 1e-6f, "relu grad pos");
        assertEquals(1f, a.grad.data[4], 1e-6f, "relu grad pos");
        assertEquals(1f, a.grad.data[5], 1e-6f, "relu grad pos");
    }

    @Test
    void testSigmoid() {
        Tensor a = Torch.tensor(new float[] { 0, 1, -1 }, 3);
        a.requires_grad = true;
        Tensor b = Torch.sigmoid(a);
        Tensor loss = Torch.sumTensor(b);
        loss.backward();

        // sig(0) = 0.5, grad = 0.5 * 0.5 = 0.25
        assertEquals(0.25f, a.grad.data[0], 1e-4f, "sigmoid grad at 0");

        // sig(1) = 0.73105, grad = sig*(1-sig) = 0.1966
        float sig1 = (float) (1.0 / (1.0 + Math.exp(-1)));
        assertEquals(sig1 * (1 - sig1), a.grad.data[1], 1e-4f, "sigmoid grad at 1");
    }

    @Test
    void testTanh() {
        Tensor a = Torch.tensor(new float[] { 0, 1, -1 }, 3);
        a.requires_grad = true;
        Tensor b = Torch.tanh(a);
        Tensor loss = Torch.sumTensor(b);
        loss.backward();

        // tanh(0) = 0, grad = 1 - 0^2 = 1
        assertEquals(1.0f, a.grad.data[0], 1e-4f, "tanh grad at 0");

        // tanh(1) = 0.76159, grad = 1 - tanh^2 = 0.41997
        float t1 = (float) Math.tanh(1);
        assertEquals(1 - t1 * t1, a.grad.data[1], 1e-4f, "tanh grad at 1");
    }
}
