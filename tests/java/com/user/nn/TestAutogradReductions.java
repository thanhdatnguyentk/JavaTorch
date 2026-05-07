package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradReductions {

    @Test
    void testSumTensor() {
        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        a.requires_grad = true;
        Tensor sumA = Torch.sumTensor(a);
        sumA.backward();

        assertNotNull(a.grad, "grad should not be null");
        assertEquals(2, a.grad.dim(), "grad dim check");
        for (int i = 0; i < 4; i++) {
            assertEquals(1f, a.grad.data[i], 1e-6f, "sum elements grad should be 1f");
        }
    }

    @Test
    void testMeanTensor() {
        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        a.requires_grad = true;
        Tensor meanA = Torch.meanTensor(a);
        meanA.backward();

        assertNotNull(a.grad, "grad should not be null");
        assertEquals(2, a.grad.dim(), "grad dim check");
        for (int i = 0; i < 4; i++) {
            assertEquals(0.25f, a.grad.data[i], 1e-6f, "mean elements grad should be 1/N (1/4=0.25)");
        }
    }
}
