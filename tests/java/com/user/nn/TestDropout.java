package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestDropout {

    @Test
    void testDropoutEval() {
        Dropout dropout = new Dropout(0.5f);
        dropout.eval();
        
        Tensor x = Torch.ones(10, 10);
        Tensor out = dropout.forward(x);
        
        for(int i=0; i<x.data.length; i++) {
            assertEquals(x.data[i], out.data[i], "Dropout in eval mode must be identity");
        }
    }

    @Test
    void testDropoutTrain() {
        Dropout dropout = new Dropout(0.5f);
        dropout.train(); // default is true
        
        // Large tensor to get stable statistics
        int n = 10000;
        Tensor x = Torch.ones(n);
        Tensor out = dropout.forward(x);
        
        int zeroCount = 0;
        float sum = 0;
        for(float v : out.data) {
            if (v == 0) {
                zeroCount++;
            } else {
                assertEquals(2.0f, v, 1e-6f, "Expected inverted scaling factor 1/(1-p) = 2.0");
            }
            sum += v;
        }
        
        float zeroRate = (float)zeroCount / n;
        assertTrue(zeroRate > 0.4 && zeroRate < 0.6, "Zero rate " + zeroRate + " should be roughly 0.5");
        
        float avg = sum / n;
        assertTrue(avg > 0.9 && avg < 1.1, "Average " + avg + " should be roughly 1.0 due to inverted scaling");
    }

    @Test
    void testDropoutGrad() {
        Dropout dropout = new Dropout(0.5f);
        dropout.train();
        
        Tensor x = Torch.ones(100);
        x.requires_grad = true;
        Tensor out = dropout.forward(x);
        
        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        
        assertNotNull(x.grad, "Gradient should be populated");
        for(int i=0; i<x.data.length; i++) {
            if (out.data[i] == 0) {
                assertEquals(0f, x.grad.data[i], "Gradient should be 0 where output was dropped");
            } else {
                // scale = 1 / (1 - 0.5) = 2.0
                assertEquals(2.0f, x.grad.data[i], 1e-6f, "Gradient should be scaled by 1/(1-p)");
            }
        }
    }

    @Test
    void testDropoutFunctional() {
        Tensor x = Torch.ones(10, 10);
        
        // Test training mode
        Tensor outTrain = Functional.dropout(x, 0.5f, true);
        float sumTrain = 0;
        for(float v : outTrain.data) sumTrain += v;
        // With p=0.5, inverted scaling is 2.0. Expected sum is roughly 100.
        // If we sum all elements, some are 2.0, some are 0.0.
        assertTrue(sumTrain > 50.0f && sumTrain < 150.0f, "Functional dropout training sum should be around 100");
        
        // Test eval mode
        Tensor outEval = Functional.dropout(x, 0.5f, false);
        float sumEval = 0;
        for(float v : outEval.data) sumEval += v;
        assertEquals(100.0f, sumEval, 1e-6f, "Functional dropout eval mode should be identity");
    }
}
