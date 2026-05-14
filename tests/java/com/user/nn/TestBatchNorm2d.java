package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.norm.BatchNorm2d;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for BatchNorm2d CPU fallback.
 * 
 * BatchNorm2d previously forced GPU via `x.toGPU()` unconditionally,
 * crashing on machines without CUDA. These tests verify the CPU path.
 */
public class TestBatchNorm2d {

    @Test
    void testBatchNorm2dCpuForwardTraining() {
        Torch.manual_seed(42);
        int C = 3;
        BatchNorm2d bn = new BatchNorm2d(C);
        bn.train();

        // Input: [N=2, C=3, H=4, W=4]
        Tensor x = Torch.randn(new int[]{2, C, 4, 4});
        Tensor out = bn.forward(x);

        // Output shape must match input shape
        assertArrayEquals(new int[]{2, C, 4, 4}, out.shape, 
            "BatchNorm2d output shape must match input shape");

        // After BN in training mode, each channel should be approximately normalized
        // (mean ≈ beta=0, std ≈ gamma=1 for default params)
        int N = 2, H = 4, W = 4;
        int spatialSize = H * W;
        for (int c = 0; c < C; c++) {
            float sum = 0f;
            for (int n = 0; n < N; n++) {
                int baseIdx = ((n * C) + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    sum += out.data[baseIdx + s];
                }
            }
            float mean = sum / (N * spatialSize);
            assertEquals(0f, mean, 1e-4, 
                "Channel " + c + " mean should be ~0 after BN training forward");
        }
    }

    @Test
    void testBatchNorm2dCpuForwardInference() {
        Torch.manual_seed(42);
        int C = 2;
        BatchNorm2d bn = new BatchNorm2d(C);

        // Run training pass first to update running stats
        bn.train();
        Tensor x1 = Torch.randn(new int[]{4, C, 3, 3});
        bn.forward(x1);

        // Switch to eval mode
        bn.eval();
        Tensor x2 = Torch.randn(new int[]{2, C, 3, 3});
        Tensor out = bn.forward(x2);

        assertArrayEquals(new int[]{2, C, 3, 3}, out.shape,
            "BatchNorm2d eval output shape must match input shape");
        
        // Verify output is not all zeros (running stats should have been updated)
        boolean anyNonZero = false;
        for (float v : out.data) {
            if (Math.abs(v) > 1e-8) {
                anyNonZero = true;
                break;
            }
        }
        assertTrue(anyNonZero, "BatchNorm2d eval output should not be all zeros");
    }

    @Test
    void testBatchNorm2dRunningStatsUpdate() {
        Torch.manual_seed(42);
        int C = 2;
        BatchNorm2d bn = new BatchNorm2d(C);
        bn.train();

        // Running mean should initially be 0
        for (int c = 0; c < C; c++) {
            assertEquals(0f, bn.runningMean.data[c], 1e-8,
                "Initial running mean should be 0");
        }

        // Forward pass should update running stats
        Tensor x = Torch.randn(new int[]{4, C, 3, 3});
        bn.forward(x);

        // After one forward pass, running mean should no longer be exactly 0
        // (unless data happens to have exact mean=0, which is astronomically unlikely)
        boolean runningMeanChanged = false;
        for (int c = 0; c < C; c++) {
            if (Math.abs(bn.runningMean.data[c]) > 1e-8) {
                runningMeanChanged = true;
            }
        }
        assertTrue(runningMeanChanged, "Running mean should change after training forward");
    }

    @Test
    void testBatchNorm2dChannelMismatch() {
        BatchNorm2d bn = new BatchNorm2d(3);
        Tensor x = Torch.randn(new int[]{1, 5, 4, 4}); // 5 channels != 3
        assertThrows(IllegalArgumentException.class, () -> bn.forward(x),
            "Should throw on channel mismatch");
    }

    @Test
    void testBatchNorm2dWrongDimensionality() {
        BatchNorm2d bn = new BatchNorm2d(3);
        Tensor x2d = Torch.randn(new int[]{4, 3}); // 2D, not 4D
        assertThrows(IllegalArgumentException.class, () -> bn.forward(x2d),
            "Should throw on non-4D input");
    }

    @Test
    void testBatchNorm2dParameters() {
        int C = 4;
        BatchNorm2d bn = new BatchNorm2d(C);
        
        // Should have 2 parameters: gamma and beta
        assertEquals(2, bn.parameters().size(),
            "BatchNorm2d should have gamma + beta parameters");
        
        // Gamma should be initialized to 1
        Tensor gamma = bn.gamma.getTensor();
        for (int i = 0; i < C; i++) {
            assertEquals(1f, gamma.data[i], 1e-6,
                "Gamma should be initialized to 1");
        }
        
        // Beta should be initialized to 0
        Tensor beta = bn.beta.getTensor();
        for (int i = 0; i < C; i++) {
            assertEquals(0f, beta.data[i], 1e-6,
                "Beta should be initialized to 0");
        }
    }

    @Test
    void testBatchNorm2dTrainEvalModes() {
        Torch.manual_seed(42);
        int C = 2;
        BatchNorm2d bn = new BatchNorm2d(C);

        // Training forward
        bn.train();
        assertTrue(bn.is_training(), "Should be in training mode");

        Tensor x = Torch.randn(new int[]{4, C, 3, 3});
        Tensor trainOut = bn.forward(x);

        // Switch to eval
        bn.eval();
        assertFalse(bn.is_training(), "Should be in eval mode");

        Tensor evalOut = bn.forward(x);

        // Training and eval outputs should generally differ
        // (because training normalizes with batch stats, eval uses running stats)
        boolean differ = false;
        for (int i = 0; i < trainOut.data.length; i++) {
            if (Math.abs(trainOut.data[i] - evalOut.data[i]) > 1e-4) {
                differ = true;
                break;
            }
        }
        assertTrue(differ, "Training and eval outputs should differ (different normalization stats)");
    }

    @Test
    void testBatchNorm2dBackwardCPU() {
        Torch.manual_seed(42);
        int C = 2;
        BatchNorm2d bn = new BatchNorm2d(C);
        bn.train();

        // Enable grad for input
        Tensor x = Torch.randn(new int[]{2, C, 3, 3});
        x.requires_grad = true;
        bn.gamma.getTensor().requires_grad = true;
        bn.beta.getTensor().requires_grad = true;

        Tensor out = bn.forward(x);
        assertTrue(out.requires_grad, "Output should require grad");

        // Sum and backward
        Tensor loss = Torch.sum_tensor(out);
        loss.backward();

        // Input gradient should exist and have correct shape
        assertNotNull(x.grad, "Input gradient should not be null");
        assertArrayEquals(x.shape, x.grad.shape,
            "Input gradient shape should match input shape");

        // Gamma gradient should exist
        assertNotNull(bn.gamma.getTensor().grad, "Gamma gradient should not be null");
        assertEquals(C, bn.gamma.getTensor().grad.numel(),
            "Gamma gradient should have C elements");

        // Beta gradient should exist
        assertNotNull(bn.beta.getTensor().grad, "Beta gradient should not be null");
        assertEquals(C, bn.beta.getTensor().grad.numel(),
            "Beta gradient should have C elements");
    }
}
