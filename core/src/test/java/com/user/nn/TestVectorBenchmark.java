package com.user.nn;

import com.user.nn.core.*;

public class TestVectorBenchmark {

    public static void main(String[] args) {
        System.out.println("=== Benchmark MATMUL ===");

        int size = 1024;
        Tensor A = Torch.randn(new int[] { size, size });
        Tensor B = Torch.randn(new int[] { size, size });

        // Warmup JIT
        System.out.println("Warming up JIT...");
        for (int i = 0; i < 3; i++) {
            Torch.matmul(A, B);
        }

        // Benchmark
        int runs = 10;
        System.out.println("Running " + runs + " iterations of MatMul " + size + "x" + size + "...");
        long start = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) {
            Torch.matmul(A, B);
        }
        long end = System.currentTimeMillis();

        System.out.printf("Avg Time per Matmul: %.2f ms%n", (end - start) / (float) runs);
    }
}
