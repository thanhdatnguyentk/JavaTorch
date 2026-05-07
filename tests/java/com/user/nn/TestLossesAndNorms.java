package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.norm.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestLossesAndNorms {

    @Test
    void testMSELoss() {
        NN.Mat a = NN.mat_alloc(2, 2);
        NN.Mat b = NN.mat_alloc(2, 2);
        a.es[0] = 1; a.es[1] = 2; a.es[2] = 3; a.es[3] = 4;
        b.es[0] = 1; b.es[1] = 1; b.es[2] = 1; b.es[3] = 1;
        float mse = Functional.mse_loss(a, b);
        // manual: ((0^2)+(1^2)+(2^2)+(3^2))/4 = (0+1+4+9)/4 = 14/4 = 3.5
        assertEquals(3.5f, mse, 1e-6f, "MSELoss mismatch");
    }

    @Test
    void testCrossEntropyLogits() {
        NN.Mat logits = NN.mat_alloc(2, 3);
        // sample logits
        logits.es[0] = 2; logits.es[1] = 1; logits.es[2] = 0;
        logits.es[3] = 0; logits.es[4] = 1; logits.es[5] = 2;
        int[] targets = new int[]{0, 2};
        float ce = Functional.cross_entropy_logits(logits, targets);
        
        // compute manually
        double denom = Math.exp(2) + Math.exp(1) + Math.exp(0);
        double logsum = Math.log(denom);
        float expected = (float) ((logsum - 2) + (logsum - 2)) / 2f;
        assertEquals(expected, ce, 1e-6f, "CrossEntropy mismatch");
    }

    @Test
    void testBatchNorm1d() {
        // BatchNorm1d: ensure output mean approx 0 per feature
        BatchNorm1d bn = new BatchNorm1d(3, false);
        NN.Mat x = NN.mat_alloc(4, 3);
        // fill with increasing values so each column has distinct mean
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.cols; j++) {
                x.es[i * x.cols + j] = i + j;
            }
        }
        
        NN.Mat out = bn.forward(x);
        
        // compute mean of out per column
        for (int j = 0; j < out.cols; j++) {
            float s = 0f;
            for (int i = 0; i < out.rows; i++) {
                s += out.es[i * out.cols + j];
            }
            float mean = s / out.rows;
            assertEquals(0.0f, mean, 1e-5f, "BatchNorm mean for column " + j + " should be zero");
        }
    }
}
