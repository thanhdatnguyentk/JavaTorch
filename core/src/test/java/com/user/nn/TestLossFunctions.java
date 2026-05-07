package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestLossFunctions {

    @Test
    void testNLLLoss() {
        // logProbs: batch=2, classes=3
        Tensor logProbs = Torch.tensor(new float[] {
                -0.5f, -1.2f, -2.0f, // sample 0
                -1.5f, -0.3f, -1.8f // sample 1
        }, 2, 3);
        logProbs.requires_grad = true;
        int[] targets = { 0, 1 }; // target class for each sample

        Tensor loss = Functional.nll_loss(logProbs, targets);
        // Expected: (-(-0.5) + -(-0.3)) / 2 = (0.5 + 0.3) / 2 = 0.4
        assertEquals(0.4f, loss.data[0], 1e-5f, "NLL loss value mismatch");

        loss.backward();
        assertNotNull(logProbs.grad, "NLL grad exists");
        // grad[0][0] = -1/2, grad[1][1] = -1/2, rest = 0
        assertEquals(-0.5f, logProbs.grad.data[0], 1e-5f, "NLL grad[0][0]");
        assertEquals(0f, logProbs.grad.data[1], 1e-5f, "NLL grad[0][1]");
        assertEquals(-0.5f, logProbs.grad.data[4], 1e-5f, "NLL grad[1][1]");
    }

    @Test
    void testMSELossTensor() {
        Tensor pred = Torch.tensor(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
        pred.requires_grad = true;
        Tensor target = Torch.tensor(new float[] { 1f, 1f, 1f, 1f }, 2, 2);

        Tensor loss = Functional.mse_loss_tensor(pred, target);
        // MSE = ((0)^2 + (1)^2 + (2)^2 + (3)^2) / 4 = (0+1+4+9)/4 = 3.5
        assertEquals(3.5f, loss.data[0], 1e-5f, "MSE loss value mismatch");

        loss.backward();
        assertNotNull(pred.grad, "MSE grad exists");
        // grad[i] = 2*(pred[i]-target[i])/n = 2*[0,1,2,3]/4 = [0, 0.5, 1.0, 1.5]
        assertEquals(0f, pred.grad.data[0], 1e-5f, "MSE grad[0]");
        assertEquals(0.5f, pred.grad.data[1], 1e-5f, "MSE grad[1]");
        assertEquals(1.0f, pred.grad.data[2], 1e-5f, "MSE grad[2]");
        assertEquals(1.5f, pred.grad.data[3], 1e-5f, "MSE grad[3]");
    }

    @Test
    void testHuberLoss() {
        Tensor pred = Torch.tensor(new float[] { 0f, 2f, 5f }, 1, 3);
        pred.requires_grad = true;
        Tensor target = Torch.tensor(new float[] { 0f, 0f, 0f }, 1, 3);
        float delta = 1f;

        Tensor loss = Functional.huber_loss(pred, target, delta);
        // errors: [0, 2, 5]
        // |0|<=1: 0.5*0=0; |2|>1: 1*(2-0.5)=1.5; |5|>1: 1*(5-0.5)=4.5
        // total = (0 + 1.5 + 4.5)/3 = 2.0
        assertEquals(2.0f, loss.data[0], 1e-5f, "Huber loss value mismatch");

        loss.backward();
        assertNotNull(pred.grad, "Huber grad exists");
        // grad[0]=0/3=0, grad[1]=delta*sign(2)/3=1/3, grad[2]=delta*sign(5)/3=1/3
        assertEquals(0f, pred.grad.data[0], 1e-5f, "Huber grad[0]");
        float expected = 1f / 3f;
        assertEquals(expected, pred.grad.data[1], 1e-5f, "Huber grad[1]");
        assertEquals(expected, pred.grad.data[2], 1e-5f, "Huber grad[2]");
    }
}
