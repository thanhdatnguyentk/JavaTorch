package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAllGPUOps {

    @Test
    public void testGPUOps() {
        if (!CUDAOps.isAvailable()) {
            System.out.println("CUDA not available, skipping test.");
            return;
        }

        try (MemoryScope scope = new MemoryScope()) {
            // 1. Test div
            System.out.println("Testing div...");
            Tensor a = Torch.tensor(new float[]{10, 20, 30}, 3).toGPU();
            Tensor b = Torch.tensor(new float[]{2, 5, 10}, 3).toGPU();
            Tensor c = Torch.div(a, b);
            
            assertTrue(c.isGPU(), "div result should be on GPU");
            c.toCPU();
            assertArrayEquals(new float[]{5.0f, 4.0f, 3.0f}, c.data, 1e-5f);
            System.out.println("div OK");

            // 2. Test pow (tensor)
            System.out.println("Testing pow (tensor)...");
            Tensor d = Torch.tensor(new float[]{2, 3, 4}, 3).toGPU();
            Tensor e = Torch.tensor(new float[]{3, 2, 0.5f}, 3).toGPU();
            Tensor f = Torch.pow(d, e);
            
            assertTrue(f.isGPU(), "pow(tensor) result should be on GPU");
            f.toCPU();
            assertArrayEquals(new float[]{8.0f, 9.0f, 2.0f}, f.data, 1e-5f);
            System.out.println("pow (tensor) OK");

            // 3. Test pow (scalar)
            System.out.println("Testing pow (scalar)...");
            Tensor g = Torch.tensor(new float[]{4, 9, 16}, 3).toGPU();
            Tensor h = Torch.pow(g, 0.5f);
            
            assertTrue(h.isGPU(), "pow(scalar) result should be on GPU");
            h.toCPU();
            assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, h.data, 1e-5f);
            System.out.println("pow (scalar) OK");

            // 4. Test sqrt
            System.out.println("Testing sqrt...");
            Tensor i = Torch.tensor(new float[]{25, 36, 49}, 3).toGPU();
            Tensor j = Torch.sqrt(i);
            
            assertTrue(j.isGPU(), "sqrt result should be on GPU");
            j.toCPU();
            assertArrayEquals(new float[]{5.0f, 6.0f, 7.0f}, j.data, 1e-5f);
            System.out.println("sqrt OK");

            // 5. Test MaxPool2d
            System.out.println("Testing MaxPool2d...");
            Tensor x_pool = Torch.tensor(new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            }, 1, 1, 3, 3).toGPU();
            com.user.nn.pooling.MaxPool2d maxpool = new com.user.nn.pooling.MaxPool2d(2, 2, 1, 1, 0, 0, 1, 3, 3);
            Tensor y_max = maxpool.forward(x_pool);
            assertTrue(y_max.isGPU(), "MaxPool2d result should be on GPU");
            y_max.toCPU();
            assertArrayEquals(new float[]{5, 6, 8, 9}, y_max.data, 1e-5f);
            System.out.println("MaxPool2d OK");

            // 6. Test AvgPool2d
            System.out.println("Testing AvgPool2d...");
            com.user.nn.pooling.AvgPool2d avgpool = new com.user.nn.pooling.AvgPool2d(2, 2, 1, 1, 0, 0, 1, 3, 3);
            Tensor y_avg = avgpool.forward(x_pool);
            assertTrue(y_avg.isGPU(), "AvgPool2d result should be on GPU");
            y_avg.toCPU();
            // (1+2+4+5)/4 = 3, (2+3+5+6)/4 = 4, (4+5+7+8)/4 = 6, (5+6+8+9)/4 = 7
            assertArrayEquals(new float[]{3, 4, 6, 7}, y_avg.data, 1e-5f);
            System.out.println("AvgPool2d OK");
            
            System.out.println("All GPU Ops verified successfully!");
        }
    }
}
