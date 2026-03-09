package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestAutogradActivations {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradActivations...");

        boolean allPassed = true;
        allPassed &= testRelu();
        allPassed &= testSigmoid();
        allPassed &= testTanh();

        if (allPassed) {
            System.out.println("TestAutogradActivations PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestAutogradActivations FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean condition, String msg) {
        if (!condition) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    // helper for fuzzy float match
    private static void checkNear(float actual, float expected, String msg) {
        if (Math.abs(actual - expected) > 1e-4) {
            System.out.println("FAIL: " + msg + " | Expected: " + expected + ", Got: " + actual);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testRelu() {
        try {
            Tensor a = Torch.tensor(new float[] { -2, -1, 0, 1, 2, 3 }, 2, 3);
            a.requires_grad = true;
            Tensor b = Torch.relu(a);
            Tensor loss = Torch.sumTensor(b);
            loss.backward();

            check(a.grad != null, "grad present");
            // derivative of ReLU is 0 if x <= 0, else 1
            check(a.grad.data[0] == 0f, "relu grad neg");
            check(a.grad.data[1] == 0f, "relu grad neg");
            check(a.grad.data[2] == 0f, "relu grad zero");
            check(a.grad.data[3] == 1f, "relu grad pos");
            check(a.grad.data[4] == 1f, "relu grad pos");
            check(a.grad.data[5] == 1f, "relu grad pos");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testSigmoid() {
        try {
            Tensor a = Torch.tensor(new float[] { 0, 1, -1 }, 3);
            a.requires_grad = true;
            Tensor b = Torch.sigmoid(a);
            Tensor loss = Torch.sumTensor(b);
            loss.backward();

            // sig(0) = 0.5, grad = 0.5 * 0.5 = 0.25
            checkNear(a.grad.data[0], 0.25f, "sigmoid grad at 0");

            // sig(1) = 0.73105, grad = sig*(1-sig) = 0.1966
            float sig1 = (float) (1.0 / (1.0 + Math.exp(-1)));
            checkNear(a.grad.data[1], sig1 * (1 - sig1), "sigmoid grad at 1");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testTanh() {
        try {
            Tensor a = Torch.tensor(new float[] { 0, 1, -1 }, 3);
            a.requires_grad = true;
            Tensor b = Torch.tanh(a);
            Tensor loss = Torch.sumTensor(b);
            loss.backward();

            // tanh(0) = 0, grad = 1 - 0^2 = 1
            checkNear(a.grad.data[0], 1.0f, "tanh grad at 0");

            // tanh(1) = 0.76159, grad = 1 - tanh^2 = 0.41997
            float t1 = (float) Math.tanh(1);
            checkNear(a.grad.data[1], 1 - t1 * t1, "tanh grad at 1");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
