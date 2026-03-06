package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.metrics.*;

public class TestMetrics {
    public static void main(String[] args) {
        System.out.println("Running TestMetrics...");

        testAccuracy();
        testMSE();
        testMAE();

        System.out.println("TestMetrics PASSED.");
    }

    private static void check(boolean condition, String msg) {
        if (!condition) {
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static void testAccuracy() {
        Accuracy acc = new Accuracy();
        
        // Batch 1: 3 samples, 2 classes
        Tensor logits1 = Torch.tensor(new float[] {
            1.0f, 0.0f, // Pred 0, Target 0 -> Correct
            0.0f, 2.0f, // Pred 1, Target 1 -> Correct
            1.5f, 1.0f  // Pred 0, Target 1 -> Incorrect
        }, 3, 2);
        int[] targets1 = new int[] { 0, 1, 1 };
        
        acc.update(logits1, targets1);
        check(Math.abs(acc.compute() - 0.6666667f) < 1e-5, "Accuracy batch 1");

        // Batch 2: 1 sample
        Tensor logits2 = Torch.tensor(new float[] { 0.5f, 1.0f }, 1, 2); // Pred 1, Target 1 -> Correct
        int[] targets2 = new int[] { 1 };
        acc.update(logits2, targets2);
        
        // Total: 3/4 correct = 0.75
        check(acc.compute() == 0.75f, "Accuracy cumulative");

        acc.reset();
        check(acc.compute() == 0f, "Accuracy reset");
    }

    private static void testMSE() {
        MeanSquaredError mse = new MeanSquaredError();
        
        Tensor p = Torch.tensor(new float[] { 1f, 2f, 3f }, 3);
        Tensor t = Torch.tensor(new float[] { 1.5f, 2.5f, 3.5f }, 3);
        
        mse.update(p, t);
        // Errors: 0.5, 0.5, 0.5 -> Squares: 0.25, 0.25, 0.25 -> Sum: 0.75 -> Mean: 0.25
        check(mse.compute() == 0.25f, "MSE batch 1");
    }

    private static void testMAE() {
        MeanAbsoluteError mae = new MeanAbsoluteError();
        
        Tensor p = Torch.tensor(new float[] { 1f, 2f, 3f }, 3);
        Tensor t = Torch.tensor(new float[] { 1.5f, 2.5f, 3.5f }, 3);
        
        mae.update(p, t);
        // Errors: 0.5, 0.5, 0.5 -> Mean: 0.5
        check(mae.compute() == 0.5f, "MAE batch 1");
    }
}
