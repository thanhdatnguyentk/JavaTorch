package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;

public class TestVectorBenchmark {

    @Test
    @Tag("slow")
    void benchmarkMatmul() {
        System.out.println("=== Benchmark MATMUL (CPU/SIMD) ===");

        int size = 512; // Reduced for standard test run
        Tensor a = Torch.randn(new int[] { size, size });
        Tensor b = Torch.randn(new int[] { size, size });

        // Warmup JIT
        for (int i = 0; i < 3; i++) {
            Torch.matmul(a, b);
        }

        // Benchmark
        int runs = 5;
        long start = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) {
            Torch.matmul(a, b);
        }
        long end = System.currentTimeMillis();

        System.out.printf("Matmul %dx%d Avg Time: %.2f ms%n", size, size, (end - start) / (float) runs);
    }
}
