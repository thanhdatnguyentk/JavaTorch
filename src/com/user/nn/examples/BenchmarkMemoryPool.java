package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.optim.*;

public class BenchmarkMemoryPool {
    public static void main(String[] args) {
        System.out.println("=== Starting GpuMemoryPool Auto-Expand Benchmark ===");
        
        int batchSize = 64;
        int channels = 3;
        int imgSize = 32; 
        
        System.out.println("Creating synthetic deep CNN model to generate large activations...");
        com.user.nn.core.Module model = new com.user.nn.core.Module() {
            // Need a lot of activations
            Conv2d c1 = new Conv2d(channels, 64, 3, 3, imgSize, imgSize, 1, 1, true);
            Conv2d c2 = new Conv2d(64, 128, 3, 3, imgSize, imgSize, 1, 1, true);
            Conv2d c3 = new Conv2d(128, 256, 3, 3, imgSize, imgSize, 1, 1, true);
            Conv2d c4 = new Conv2d(256, 256, 3, 3, imgSize, imgSize, 1, 1, true);

            public Tensor forward(Tensor x) {
                x = c1.forward(x);
                x = Torch.relu(x);
                x = c2.forward(x);
                x = Torch.relu(x);
                x = c3.forward(x);
                x = Torch.relu(x);
                x = c4.forward(x);
                return x;
            }
        };

        model.toGPU();
        Optim.SGD optim = new Optim.SGD(model.parameters(), 0.01f);

        // Intentionally initialize with a tiny multiplier (2x params) to guarantee pool exhaustion
        System.out.println("Initializing GpuMemoryPool with tiny size to simulate large batch starvation...");
        GpuMemoryPool.autoInit(model, 2.0f); 

        System.out.println("Starting Benchmark (5 Steps)...");
        int steps = 6;
        long[] times = new long[steps];
        long[] fallbacks = new long[steps];
        
        for (int i = 0; i < steps; i++) {
            Tensor.fallbackAllocations = 0; // Reset counter for visibility per step
            long start = System.currentTimeMillis();
            
            try (MemoryScope scope = new MemoryScope()) {
                Tensor x = Torch.randn(new int[]{batchSize, channels, imgSize, imgSize}).toGPU();
                
                Tensor out = model.forward(x);
                // Use a simple mean over all activation elements as a dummy loss
                Tensor loss = Torch.mean_tensor(out);
                
                optim.zero_grad();
                loss.backward();
                optim.step();
            }
            
            long end = System.currentTimeMillis();
            times[i] = end - start;
            fallbacks[i] = Tensor.fallbackAllocations;
            
            System.out.printf(">> Step %d completed in %d ms | Fallback Allocations: %d%n", i + 1, times[i], fallbacks[i]);
            System.out.println("---------------------------------------------------------");
        }
        
        System.out.println("=== Benchmark Data (CSV) ===");
        System.out.println("Step,Latency_ms,Fallback_Count");
        for (int i = 0; i < steps; i++) {
            System.out.printf("%d,%d,%d%n", i + 1, times[i], fallbacks[i]);
        }
        System.out.println("=========================================================");
    }
}
