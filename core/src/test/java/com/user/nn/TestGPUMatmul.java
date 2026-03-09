package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import java.util.Arrays;

public class TestGPUMatmul {
    public static void main(String[] args) {
        try {
            System.out.println("=== Testing GPU Matmul ===");
            
            float[] d1 = {1, 2, 3, 4, 5, 6};
            float[] d2 = {7, 8, 9, 10, 11, 12};
            
            Tensor a = Torch.tensor(d1, 2, 3);
            Tensor b = Torch.tensor(d2, 3, 2);
            
            System.out.println("A (CPU):\n" + a);
            System.out.println("B (CPU):\n" + b);
            
            // Move to GPU
            System.out.println("Moving A and B to GPU...");
            a.toGPU();
            b.toGPU();
            System.out.println("A device: " + a.device);
            System.out.println("B device: " + b.device);
            
            // Matmul on GPU
            System.out.println("Executing Matmul on GPU...");
            Tensor c = Torch.matmul(a, b);
            System.out.println("Matmul result device: " + c.device);
            
            // Result back to CPU (toString handles this)
            System.out.println("Result:\n" + c);
            
            // Verify with CPU matmul
            System.out.println("Verifying with CPU implementation...");
            Tensor a_cpu = Torch.tensor(d1, 2, 3);
            Tensor b_cpu = Torch.tensor(d2, 3, 2);
            Tensor c_cpu = Torch.matmul(a_cpu, b_cpu);
            System.out.println("Expected (CPU):\n" + c_cpu);
            
            boolean match = true;
            for(int i = 0; i < c.data.length; i++) {
                if(Math.abs(c.data[i] - c_cpu.data[i]) > 1e-4) {
                    match = false;
                    System.out.println("Mismatch at index " + i + ": got " + c.data[i] + ", expected " + c_cpu.data[i]);
                }
            }
            
            if (match) {
                System.out.println("SUCCESS: GPU Matmul result matches CPU!");
            } else {
                System.out.println("FAILURE: GPU Matmul result mismatch.");
                System.exit(1);
            }

            // Test Autograd
            System.out.println("\n=== Testing GPU Autograd (Fallback) ===");
            a.requires_grad = true;
            b.requires_grad = true;
            
            Tensor out = Torch.matmul(a, b);
            // sumTensor uses sum(a) which we should probably fix for GPU, but let's see if it falls back to CPU
            Tensor loss = Torch.sumTensor(out);
            loss.backward();
            
            System.out.println("A grad (computed on GPU/Fallback):\n" + a.grad);
            System.out.println("B grad (computed on GPU/Fallback):\n" + b.grad);
            
            System.out.println("\n=== Testing GPU ReLU ===");
            Tensor x = Torch.tensor(new float[]{-1, 0, 1, 2}, 4);
            x.toGPU();
            Tensor y = Torch.relu(x);
            System.out.println("Input: " + x);
            System.out.println("ReLU result: " + y);
            
            boolean reluMatch = (y.data[0] == 0 && y.data[1] == 0 && y.data[2] == 1 && y.data[3] == 2);
            System.out.println("ReLU Match: " + reluMatch);
            if (!reluMatch) System.exit(1);

            // Clean up
            a.close();
            b.close();
            c.close();
            out.close();
            loss.close();
            x.close();
            y.close();
            
            System.out.println("\nGPU Matmul and ReLU tests passed!");

        } catch (Throwable e) {
            System.err.println("Test failed with exception:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
