package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

/**
 * Tests for autograd-aware loss functions: nll_loss, mse_loss_tensor,
 * huber_loss.
 */
public class TestLossFunctions {
    public static void main(String[] args) {
        System.out.println("Running TestLossFunctions...");
        boolean allPassed = true;
        allPassed &= testNLLLoss();
        allPassed &= testMSELossTensor();
        allPassed &= testHuberLoss();

        if (allPassed) {
            System.out.println("TestLossFunctions PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestLossFunctions FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testNLLLoss() {
        try {
            // logProbs: batch=2, classes=3
            // Simulating log(softmax) manually
            Tensor logProbs = Torch.tensor(new float[] {
                    -0.5f, -1.2f, -2.0f, // sample 0
                    -1.5f, -0.3f, -1.8f // sample 1
            }, 2, 3);
            logProbs.requires_grad = true;
            int[] targets = { 0, 1 }; // target class for each sample

            Tensor loss = NN.F.nll_loss(logProbs, targets);
            // Expected: (-(-0.5) + -(-0.3)) / 2 = (0.5 + 0.3) / 2 = 0.4
            check(Math.abs(loss.data[0] - 0.4f) < 1e-5f, "NLL loss value, got " + loss.data[0]);

            loss.backward();
            check(logProbs.grad != null, "NLL grad exists");
            // grad[0][0] = -1/2, grad[1][1] = -1/2, rest = 0
            check(Math.abs(logProbs.grad.data[0] - (-0.5f)) < 1e-5f, "NLL grad[0][0]");
            check(Math.abs(logProbs.grad.data[1]) < 1e-5f, "NLL grad[0][1]");
            check(Math.abs(logProbs.grad.data[4] - (-0.5f)) < 1e-5f, "NLL grad[1][1]");

            System.out.println("  nll_loss OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testMSELossTensor() {
        try {
            Tensor pred = Torch.tensor(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
            pred.requires_grad = true;
            Tensor target = Torch.tensor(new float[] { 1f, 1f, 1f, 1f }, 2, 2);

            Tensor loss = NN.F.mse_loss_tensor(pred, target);
            // MSE = ((0)^2 + (1)^2 + (2)^2 + (3)^2) / 4 = (0+1+4+9)/4 = 3.5
            check(Math.abs(loss.data[0] - 3.5f) < 1e-5f, "MSE loss value, got " + loss.data[0]);

            loss.backward();
            check(pred.grad != null, "MSE grad exists");
            // grad[i] = 2*(pred[i]-target[i])/n = 2*[0,1,2,3]/4 = [0, 0.5, 1.0, 1.5]
            check(Math.abs(pred.grad.data[0] - 0f) < 1e-5f, "MSE grad[0]");
            check(Math.abs(pred.grad.data[1] - 0.5f) < 1e-5f, "MSE grad[1]");
            check(Math.abs(pred.grad.data[2] - 1.0f) < 1e-5f, "MSE grad[2]");
            check(Math.abs(pred.grad.data[3] - 1.5f) < 1e-5f, "MSE grad[3]");

            System.out.println("  mse_loss_tensor OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testHuberLoss() {
        try {
            Tensor pred = Torch.tensor(new float[] { 0f, 2f, 5f }, 1, 3);
            pred.requires_grad = true;
            Tensor target = Torch.tensor(new float[] { 0f, 0f, 0f }, 1, 3);
            float delta = 1f;

            Tensor loss = NN.F.huber_loss(pred, target, delta);
            // errors: [0, 2, 5]
            // |0|<=1: 0.5*0=0; |2|>1: 1*(2-0.5)=1.5; |5|>1: 1*(5-0.5)=4.5
            // total = (0 + 1.5 + 4.5)/3 = 2.0
            check(Math.abs(loss.data[0] - 2.0f) < 1e-5f, "Huber loss value, got " + loss.data[0]);

            loss.backward();
            check(pred.grad != null, "Huber grad exists");
            // grad[0]=0/3=0, grad[1]=delta*sign(2)/3=1/3, grad[2]=delta*sign(5)/3=1/3
            check(Math.abs(pred.grad.data[0]) < 1e-5f, "Huber grad[0]");
            float expected = 1f / 3f;
            check(Math.abs(pred.grad.data[1] - expected) < 1e-5f, "Huber grad[1], got " + pred.grad.data[1]);
            check(Math.abs(pred.grad.data[2] - expected) < 1e-5f, "Huber grad[2], got " + pred.grad.data[2]);

            System.out.println("  huber_loss OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
