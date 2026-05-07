package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradShapeOps {

    @Test
    void testReshape() {
        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        a.requires_grad = true;
        Tensor b = a.reshape(4);
        Tensor loss = Torch.sumTensor(b);
        loss.backward();

        assertNotNull(a.grad, "grad should not be null");
        assertEquals(2, a.grad.dim(), "grad should have original rank");
        assertArrayEquals(new int[]{2, 2}, a.grad.shape, "grad shape should be 2x2");
        for (int i = 0; i < 4; i++) {
            assertEquals(1f, a.grad.data[i], 1e-6f, "grad should be 1f");
        }
    }

    @Test
    void testPermute() {
        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        a.requires_grad = true;
        Tensor b = Torch.permute(a, 1, 0); // shape: 3, 2

        Tensor mask = Torch.tensor(new float[] { 1, 0, 0, 1, 0, 0 }, 3, 2);
        Tensor loss = Torch.sumTensor(Torch.mul(b, mask));
        loss.backward();

        // b = [[1, 4], [2, 5], [3, 6]]
        // mask = [[1, 0], [0, 1], [0, 0]]
        // so a_00 (1) and a_11 (5) receive a grad of 1.
        assertEquals(1f, a.grad.data[0], 1e-6f, "a_00 grad");
        assertEquals(1f, a.grad.data[4], 1e-6f, "a_11 grad");
        assertEquals(0f, a.grad.data[1], 1e-6f, "a_01 grad");
    }

    @Test
    void testTranspose() {
        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        a.requires_grad = true;
        Tensor b = Torch.transpose(a, 0, 1);
        Tensor loss = Torch.sumTensor(b);
        loss.backward();

        assertArrayEquals(new int[]{2, 3}, a.grad.shape, "grad shape mismatch");
        for (int i = 0; i < 6; i++) {
            assertEquals(1f, a.grad.data[i], 1e-6f, "grad should be 1f");
        }
    }

    @Test
    void testSqueeze() {
        Tensor a = Torch.tensor(new float[] { 1, 2 }, 1, 2, 1);
        a.requires_grad = true;
        Tensor b = a.squeeze(); // shape: 2
        Tensor c = b.unsqueeze(1); // shape: 2, 1

        Tensor loss = Torch.sumTensor(c);
        loss.backward();

        assertEquals(3, a.grad.dim(), "grad rank mismatch");
        assertArrayEquals(new int[]{1, 2, 1}, a.grad.shape, "grad shape mismatch");
        for (int i = 0; i < 2; i++) {
            assertEquals(1f, a.grad.data[i], 1e-6f, "grad should be 1f");
        }
    }
}
