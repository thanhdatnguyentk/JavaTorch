package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.layers.Conv2d;
import com.user.nn.pooling.MaxPool2d;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

public class TestGPUBenchmark {

    @Test
    @Tag("gpu")
    @Tag("slow")
    void runBenchmarks() {
        assumeTrue(Torch.hasGPU(), "GPU not available, skipping benchmarks");
        
        System.out.println("====== GPU vs CPU Optimization Benchmark ======");
        boolean previousGradConfig = Torch.is_grad_enabled();
        Torch.set_grad_enabled(false); // Evaluate forward pass times only
        
        try {
            benchmarkMatmul(512, 10); // Reduced size/runs for CI safety
            benchmarkConv2d(16, 32, 32, 10);
            benchmarkMaxPool2d(16, 32, 32, 10);
            benchmarkReLU(1024 * 1024, 10);
        } finally {
            Torch.set_grad_enabled(previousGradConfig);
        }
        
        System.out.println("\nBenchmark completed.");
    }

    private void benchmarkMatmul(int size, int runs) {
        System.out.println("\n--- Benchmarking Matmul (" + size + "x" + size + ") ---");
        Tensor a = Torch.randn(new int[]{size, size});
        Tensor b = Torch.randn(new int[]{size, size});
        
        for (int i = 0; i < 3; i++) Torch.matmul(a, b);
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.matmul(a, b);
        long endCPU = System.currentTimeMillis();
        
        a.toGPU();
        b.toGPU();
        for (int i = 0; i < 3; i++) Torch.matmul(a, b);
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.matmul(a, b);
        long endGPU = System.currentTimeMillis();
        
        printResults("Matmul", endCPU - startCPU, endGPU - startGPU, runs);
        a.close(); b.close();
    }

    private void benchmarkConv2d(int batch, int channels, int size, int runs) {
        System.out.println("\n--- Benchmarking Conv2d (B: " + batch + ", C: " + channels + ", S: " + size + ") ---");
        Tensor x = Torch.randn(new int[]{batch, channels, size, size});
        Conv2d conv = new Conv2d(channels, 32, 3, 3, size, size, 1, 1, true);
        
        for (int i = 0; i < 2; i++) conv.forward(x);
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) conv.forward(x);
        long endCPU = System.currentTimeMillis();
        
        x.toGPU();
        conv.weight.getTensor().toGPU();
        if (conv.bias != null) conv.bias.getTensor().toGPU();
        for (int i = 0; i < 2; i++) conv.forward(x);
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) conv.forward(x);
        long endGPU = System.currentTimeMillis();
        
        printResults("Conv2d", endCPU - startCPU, endGPU - startGPU, runs);
        x.close();
    }

    private void benchmarkMaxPool2d(int batch, int channels, int size, int runs) {
        System.out.println("\n--- Benchmarking MaxPool2d (B: " + batch + ", C: " + channels + ", S: " + size + ") ---");
        Tensor x = Torch.randn(new int[]{batch, channels, size, size});
        MaxPool2d pool = new MaxPool2d(2, 2, 2, 2, 0, 0, channels, size, size);
        
        for (int i = 0; i < 2; i++) pool.forward(x);
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) pool.forward(x);
        long endCPU = System.currentTimeMillis();
        
        x.toGPU();
        for (int i = 0; i < 2; i++) pool.forward(x);
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) pool.forward(x);
        long endGPU = System.currentTimeMillis();
        
        printResults("MaxPool2d", endCPU - startCPU, endGPU - startGPU, runs);
        x.close();
    }

    private void benchmarkReLU(int size, int runs) {
        System.out.println("\n--- Benchmarking ReLU (Size: " + size + ") ---");
        Tensor x = Torch.randn(new int[]{size});
        
        for (int i = 0; i < 2; i++) Torch.relu(x);
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.relu(x);
        long endCPU = System.currentTimeMillis();
        
        x.toGPU();
        for (int i = 0; i < 2; i++) Torch.relu(x);
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.relu(x);
        long endGPU = System.currentTimeMillis();
        
        printResults("ReLU", endCPU - startCPU, endGPU - startGPU, runs);
        x.close();
    }

    private void printResults(String name, long cpuTime, long gpuTime, int runs) {
        System.out.printf("%s -> CPU Avg: %.2f ms | GPU Avg: %.2f ms | Speedup: %.2fx\n", 
                name, cpuTime / (float)runs, gpuTime / (float)runs, (float)cpuTime / Math.max(1, gpuTime));
    }
}
