package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

public class TestPhase4GPU {

    @Test
    @Tag("gpu")
    void testGPUReductions() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2).toGPU();
        a.requires_grad = true;
        
        // Test sumTensor (using standard name)
        Tensor s = Torch.sumTensor(a);
        assertTrue(s.isGPU(), "Sum result should be on GPU");
        s.backward();
        
        a.grad.toCPU();
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0f, a.grad.data[i], 1e-6f, "Sum grad mismatch at " + i);
        }
        
        // Test meanTensor
        a.zero_grad();
        Tensor m = Torch.meanTensor(a);
        m.backward();
        a.grad.toCPU();
        for (int i = 0; i < 4; i++) {
            assertEquals(0.25f, a.grad.data[i], 1e-6f, "Mean grad mismatch at " + i);
        }
        
        a.close(); s.close(); m.close();
    }

    @Test
    @Tag("gpu")
    void testGPUActivations() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        float[] data = {-1.0f, 0.0f, 1.0f, 2.0f};
        Tensor x = Torch.tensor(data, 4).toGPU();
        x.requires_grad = true;

        // Test ReLU
        Tensor y = Torch.relu(x);
        assertTrue(y.isGPU(), "ReLU output should be on GPU");
        y.backward(Torch.ones(4).toGPU());
        x.grad.toCPU();
        assertEquals(0.0f, x.grad.data[0], 1e-6f, "ReLU grad mismatch at index 0");
        assertEquals(1.0f, x.grad.data[2], 1e-6f, "ReLU grad mismatch at index 2");

        // Test LeakyReLU
        x.zero_grad();
        Tensor y2 = Torch.leaky_relu(x, 0.1f);
        y2.backward(Torch.ones(4).toGPU());
        x.grad.toCPU();
        assertEquals(0.1f, x.grad.data[0], 1e-6f, "LeakyReLU grad mismatch at index 0");

        x.close(); y.close(); y2.close();
    }

    @Test
    @Tag("gpu")
    void testGPUBCE() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        Tensor input = Torch.tensor(new float[]{0.1f, 0.9f, 0.4f}, 3).toGPU();
        Tensor target = Torch.tensor(new float[]{0.0f, 1.0f, 0.0f}, 3).toGPU();
        input.requires_grad = true;

        Tensor loss = Functional.binary_cross_entropy(input, target);
        assertTrue(loss.isGPU(), "BCE output should be on GPU");
        loss.backward();

        assertNotNull(input.grad, "BCE grad should not be null");
        input.grad.toCPU();
        // Gradient of -[y*log(h) + (1-y)*log(1-h)] is (h-y)/(h*(1-h))
        // For first element: (0.1 - 0.0) / (0.1 * 0.9) = 0.1 / 0.09 = 1.111...
        // Scaled by 1/N (1/3)
        float expectedGrad0 = (0.1f - 0.0f) / (0.1f * 0.9f) / 3.0f;
        assertEquals(expectedGrad0, input.grad.data[0], 1e-4f, "BCE grad mismatch");

        input.close(); target.close(); loss.close();
    }
}
