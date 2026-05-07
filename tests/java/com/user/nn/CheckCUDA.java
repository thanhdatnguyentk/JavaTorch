package com.user.nn;
import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;

public class CheckCUDA {
    @Test
    @Tag("gpu")
    void testCUDAAvailable() {
        System.out.println("CUDAOps.isAvailable(): " + CUDAOps.isAvailable());
        if (CUDAOps.isAvailable()) {
            Tensor t = new Tensor(10).toGPU();
            assertTrue(t.isGPU(), "Tensor should be on GPU");
            System.out.println("Tensor.isGPU(): " + t.isGPU());
        } else {
            System.out.println("CUDA not available, skipping GPU tensor check");
        }
    }
}
