package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

public class TestGPUMatmul {

    @Test
    @Tag("gpu")
    void testGPUMatmul() {
        assumeTrue(Torch.hasGPU(), "GPU not available");

        float[] d1 = {1, 2, 3, 4, 5, 6};
        float[] d2 = {7, 8, 9, 10, 11, 12};
        
        Tensor a = Torch.tensor(d1, 2, 3);
        Tensor b = Torch.tensor(d2, 3, 2);
        
        a.toGPU();
        b.toGPU();
        
        Tensor c = Torch.matmul(a, b);
        assertEquals(Tensor.Device.GPU, c.device);
        
        // Verify with CPU
        Tensor a_cpu = Torch.tensor(d1, 2, 3);
        Tensor b_cpu = Torch.tensor(d2, 3, 2);
        Tensor c_cpu = Torch.matmul(a_cpu, b_cpu);
        
        c.toCPU();
        assertArrayEquals(c_cpu.data, c.data, 1e-4f, "GPU matmul mismatch with CPU reference");
        
        a.close(); b.close(); c.close();
    }

    @Test
    @Tag("gpu")
    void testGPUAutogradFallback() {
        assumeTrue(Torch.hasGPU(), "GPU not available");

        Tensor a = Torch.tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor b = Torch.tensor(new float[]{7, 8, 9, 10, 11, 12}, 3, 2);
        a.toGPU(); b.toGPU();
        a.requires_grad = true; b.requires_grad = true;

        Tensor out = Torch.matmul(a, b);
        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(a.grad, "a.grad should not be null");
        assertNotNull(b.grad, "b.grad should not be null");
        
        a.close(); b.close(); out.close(); loss.close();
    }

    @Test
    @Tag("gpu")
    void testGPURelu() {
        assumeTrue(Torch.hasGPU(), "GPU not available");

        Tensor x = Torch.tensor(new float[]{-1, 0, 1, 2}, 4);
        x.toGPU();
        Tensor y = Torch.relu(x);
        
        y.toCPU();
        assertArrayEquals(new float[]{0, 0, 1, 2}, y.data, 1e-6f);
        
        x.close(); y.close();
    }
}
