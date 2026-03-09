package com.user.nn;

import com.user.nn.core.*;

public class TestPhase4GPU {
    public static void main(String[] args) {
        System.out.println("Running TestPhase4GPU...");
        if (!CUDAOps.isAvailable()) {
            System.out.println("CUDA not available. Skipping Phase 4 GPU tests.");
            System.exit(0);
        }

        boolean allPassed = true;
        allPassed &= testGPUReductions();
        allPassed &= testGPUActivations();
        allPassed &= testGPUBCE();

        if (allPassed) {
            System.out.println("TestPhase4GPU PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestPhase4GPU FAILED.");
            System.exit(1);
        }
    }

    private static boolean testGPUReductions() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2).toGPU();
            a.requires_grad = true;
            
            // Test sum_tensor
            Tensor s = Torch.sum_tensor(a);
            check(s.isGPU(), "Sum result should be on GPU");
            s.backward();
            
            a.grad.toCPU();
            for (int i = 0; i < 4; i++) {
                check(Math.abs(a.grad.data[i] - 1.0f) < 1e-6f, "Sum grad mismatch at " + i);
            }
            
            // Test mean_tensor
            a.zero_grad();
            Tensor m = Torch.mean_tensor(a);
            m.backward();
            a.grad.toCPU();
            for (int i = 0; i < 4; i++) {
                check(Math.abs(a.grad.data[i] - 0.25f) < 1e-6f, "Mean grad mismatch at " + i);
            }
            
            System.out.println("  GPU Reductions OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testGPUActivations() {
        try {
            float[] data = {-1.0f, 0.0f, 1.0f, 2.0f};
            Tensor x = Torch.tensor(data, 4).toGPU();
            x.requires_grad = true;

            // Test ReLU
            Tensor y = Torch.relu(x);
            check(y.isGPU(), "ReLU output should be on GPU");
            y.backward(Torch.ones(4).toGPU());
            x.grad.toCPU();
            check(x.grad.data[0] == 0.0f, "ReLU grad mismatch at 0");
            check(x.grad.data[2] == 1.0f, "ReLU grad mismatch at 2");

            // Test LeakyReLU
            x.zero_grad();
            Tensor y2 = Torch.leaky_relu(x, 0.1f);
            y2.backward(Torch.ones(4).toGPU());
            x.grad.toCPU();
            check(Math.abs(x.grad.data[0] - 0.1f) < 1e-6f, "LeakyReLU grad mismatch at 0");

            System.out.println("  GPU Activations OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testGPUBCE() {
        try {
            Tensor input = Torch.tensor(new float[]{0.1f, 0.9f, 0.4f}, 3).toGPU();
            Tensor target = Torch.tensor(new float[]{0.0f, 1.0f, 0.0f}, 3).toGPU();
            input.requires_grad = true;

            Tensor loss = Functional.binary_cross_entropy(input, target);
            check(loss.isGPU(), "BCE output should be on GPU");
            loss.backward();

            check(input.grad != null, "BCE grad should not be null");
            input.grad.toCPU();
            // Gradient of -[y*log(h) + (1-y)*log(1-h)] is (h-y)/(h*(1-h))
            // For first element: (0.1 - 0.0) / (0.1 * 0.9) = 0.1 / 0.09 = 1.111...
            // Scaled by 1/N (1/3)
            float expectedGrad0 = (0.1f - 0.0f) / (0.1f * 0.9f) / 3.0f;
            check(Math.abs(input.grad.data[0] - expectedGrad0) < 1e-4f, "BCE grad mismatch, got " + input.grad.data[0]);

            System.out.println("  GPU BCE Loss OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }
}
