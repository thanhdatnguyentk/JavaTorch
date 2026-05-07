package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.metrics.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestMetrics {

    @Test
    void testAccuracy() {
        Accuracy acc = new Accuracy();
        
        // Batch 1: 3 samples, 2 classes
        Tensor logits1 = Torch.tensor(new float[] {
            1.0f, 0.0f, // Pred 0, Target 0 -> Correct
            0.0f, 2.0f, // Pred 1, Target 1 -> Correct
            1.5f, 1.0f  // Pred 0, Target 1 -> Incorrect
        }, 3, 2);
        int[] targets1 = new int[] { 0, 1, 1 };
        
        acc.update(logits1, targets1);
        assertEquals(2.0f/3.0f, acc.compute(), 1e-5f, "Accuracy batch 1 mismatch");

        // Batch 2: 1 sample
        Tensor logits2 = Torch.tensor(new float[] { 0.5f, 1.0f }, 1, 2); // Pred 1, Target 1 -> Correct
        int[] targets2 = new int[] { 1 };
        acc.update(logits2, targets2);
        
        // Total: 3/4 correct = 0.75
        assertEquals(0.75f, acc.compute(), 1e-6f, "Accuracy cumulative mismatch");

        acc.reset();
        assertEquals(0.0f, acc.compute(), "Accuracy reset failed");
    }

    @Test
    void testMSE() {
        MeanSquaredError mse = new MeanSquaredError();
        
        Tensor p = Torch.tensor(new float[] { 1f, 2f, 3f }, 3);
        Tensor t = Torch.tensor(new float[] { 1.5f, 2.5f, 3.5f }, 3);
        
        mse.update(p, t);
        // Errors: 0.5, 0.5, 0.5 -> Squares: 0.25, 0.25, 0.25 -> Sum: 0.75 -> Mean: 0.25
        assertEquals(0.25f, mse.compute(), 1e-6f, "MSE mismatch");
    }

    @Test
    void testMAE() {
        MeanAbsoluteError mae = new MeanAbsoluteError();
        
        Tensor p = Torch.tensor(new float[] { 1f, 2f, 3f }, 3);
        Tensor t = Torch.tensor(new float[] { 1.5f, 2.5f, 3.5f }, 3);
        
        mae.update(p, t);
        // Errors: 0.5, 0.5, 0.5 -> Mean: 0.5
        assertEquals(0.5f, mae.compute(), 1e-6f, "MAE mismatch");
    }
}
