package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.core.NN;

public class TestGPUBenchmark {

    private static void benchmarkMatmul(int size, int runs) {
        System.out.println("\n--- Benchmarking Matmul (" + size + "x" + size + ") ---");
        Tensor a = Torch.randn(new int[]{size, size});
        Tensor b = Torch.randn(new int[]{size, size});
        
        // Warmup CPU
        for (int i = 0; i < 3; i++) Torch.matmul(a, b);
        
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.matmul(a, b);
        long endCPU = System.currentTimeMillis();
        
        // Move to GPU
        a.toGPU();
        b.toGPU();
        
        // Warmup GPU
        for (int i = 0; i < 3; i++) Torch.matmul(a, b);
        
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.matmul(a, b);
        long endGPU = System.currentTimeMillis();
        
        System.out.printf("CPU (SIMD) Avg Time: %.2f ms\n", (endCPU - startCPU) / (float)runs);
        System.out.printf("GPU Avg Time: %.2f ms\n", (endGPU - startGPU) / (float)runs);
        System.out.printf("Speedup: %.2fx\n", (float)(endCPU - startCPU) / Math.max(1, endGPU - startGPU));
        a.close(); b.close();
    }

    private static void benchmarkConv2d(NN nn, int batch, int channels, int size, int runs) {
        System.out.println("\n--- Benchmarking Conv2d (Batch: " + batch + ", Channels: " + channels + ", Size: " + size + "x" + size + ") ---");
        Tensor x = Torch.randn(new int[]{batch, channels, size, size});
        // NN outer, inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW, bias
        NN.Conv2d conv = new NN.Conv2d(nn, channels, 32, 3, 3, 1, 1, 1, 1, true);
        
        // Warmup CPU
        for (int i = 0; i < 3; i++) conv.forward(x);
        
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) conv.forward(x);
        long endCPU = System.currentTimeMillis();
        
        // Move to GPU
        x.toGPU();
        conv.weight.getTensor().toGPU();
        if (conv.bias != null) conv.bias.getTensor().toGPU();
        
        // Warmup GPU
        for (int i = 0; i < 3; i++) conv.forward(x);
        
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) conv.forward(x);
        long endGPU = System.currentTimeMillis();
        
        System.out.printf("CPU Avg Time: %.2f ms\n", (endCPU - startCPU) / (float)runs);
        System.out.printf("GPU Avg Time: %.2f ms\n", (endGPU - startGPU) / (float)runs);
        System.out.printf("Speedup: %.2fx\n", (float)(endCPU - startCPU) / Math.max(1, endGPU - startGPU));
        x.close();
    }
    
    private static void benchmarkMaxPool2d(NN nn, int batch, int channels, int size, int runs) {
        System.out.println("\n--- Benchmarking MaxPool2d (Batch: " + batch + ", Channels: " + channels + ", Size: " + size + "x" + size + ") ---");
        Tensor x = Torch.randn(new int[]{batch, channels, size, size});
        // kernelH, kernelW, strideH, strideW, padH, padW, inC, inH, inW
        NN.MaxPool2d pool = new NN.MaxPool2d(2, 2, 2, 2, 0, 0, channels, size, size);
        
        // Warmup CPU
        for (int i = 0; i < 3; i++) pool.forward(x);
        
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) pool.forward(x);
        long endCPU = System.currentTimeMillis();
        
        // Move to GPU
        x.toGPU();
        
        // Warmup GPU
        for (int i = 0; i < 3; i++) pool.forward(x);
        
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) pool.forward(x);
        long endGPU = System.currentTimeMillis();
        
        System.out.printf("CPU Avg Time: %.2f ms\n", (endCPU - startCPU) / (float)runs);
        System.out.printf("GPU Avg Time: %.2f ms\n", (endGPU - startGPU) / (float)runs);
        System.out.printf("Speedup: %.2fx\n", (float)(endCPU - startCPU) / Math.max(1, endGPU - startGPU));
        x.close();
    }
    
    private static void benchmarkReLU(int size, int runs) {
        System.out.println("\n--- Benchmarking ReLU (Size: " + size + ") ---");
        Tensor x = Torch.randn(new int[]{size});
        
        // Warmup CPU
        for (int i = 0; i < 3; i++) Torch.relu(x);
        
        long startCPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.relu(x);
        long endCPU = System.currentTimeMillis();
        
        // Move to GPU
        x.toGPU();
        
        // Warmup GPU
        for (int i = 0; i < 3; i++) Torch.relu(x);
        
        long startGPU = System.currentTimeMillis();
        for (int i = 0; i < runs; i++) Torch.relu(x);
        long endGPU = System.currentTimeMillis();
        
        System.out.printf("CPU Avg Time: %.2f ms\n", (endCPU - startCPU) / (float)runs);
        System.out.printf("GPU Avg Time: %.2f ms\n", (endGPU - startGPU) / (float)runs);
        System.out.printf("Speedup: %.2fx\n", (float)(endCPU - startCPU) / Math.max(1, endGPU - startGPU));
        x.close();
    }

    public static void main(String[] args) {
        System.out.println("====== GPU vs CPU Optimization Benchmark ======");
        try {
            boolean previousGradConfig = Torch.is_grad_enabled();
            Torch.set_grad_enabled(false); // Evaluate forward pass times only
            
            NN nn = new NN();
            
            // Benchmark Matrix Multiplication (CPU uses SIMD Vector API)
            benchmarkMatmul(1024, 20); // 1K x 1K matrix
            System.gc();
            
            // Benchmark Convolution
            benchmarkConv2d(nn, 32, 64, 64, 20); // 32 batch, 64 channels, 64x64 img
            System.gc();
            
            // Benchmark Max Pooling
            benchmarkMaxPool2d(nn, 32, 64, 64, 20); 
            System.gc();
            
            // Benchmark ReLU (element-wise)
            benchmarkReLU(1024 * 1024 * 10, 20); // 10 million elements
            System.gc();
            
            Torch.set_grad_enabled(previousGradConfig);
            
            System.out.println("\nBenchmark completed successfully.");
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(1);
        }
    }
}
