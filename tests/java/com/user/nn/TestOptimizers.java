package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

/**
 * Tests for Optim.SGD and Optim.Adam on simple quadratic minimization.
 * f(x) = sum(x^2), gradient = 2*x, minimum at x=0.
 */
public class TestOptimizers {
    public static void main(String[] args) {
        System.out.println("Running TestOptimizers...");
        boolean allPassed = true;
        allPassed &= testSGDNoMomentum();
        allPassed &= testSGDWithMomentum();
        allPassed &= testAdam();

        if (allPassed) {
            System.out.println("TestOptimizers PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestOptimizers FAILED.");
            System.exit(1);
        }
    }

    // Minimize f(x) = x0^2 + x1^2 with plain SGD
    private static boolean testSGDNoMomentum() {
        try {
            NN.Linear layer = new NN.Linear(new NN(), 2, 1, false); // no bias
            Tensor w = layer.weight.getTensor();
            w.data[0] = 5f;
            w.data[1] = -3f;
            w.requires_grad = true;

            Optim.SGD opt = new Optim.SGD(layer.parameters(), 0.1f);

            for (int i = 0; i < 100; i++) {
                opt.zero_grad();
                // f = w[0]^2 + w[1]^2
                Tensor loss = Torch.sumTensor(Torch.mul(w, w));
                loss.backward();
                opt.step();
            }
            // After 100 steps, should converge near 0
            check(Math.abs(w.data[0]) < 0.01f, "SGD w[0] near 0, got " + w.data[0]);
            check(Math.abs(w.data[1]) < 0.01f, "SGD w[1] near 0, got " + w.data[1]);
            System.out.println("  SGD (no momentum) OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // Minimize with SGD + momentum
    private static boolean testSGDWithMomentum() {
        try {
            NN.Linear layer = new NN.Linear(new NN(), 2, 1, false);
            Tensor w = layer.weight.getTensor();
            w.data[0] = 5f;
            w.data[1] = -3f;
            w.requires_grad = true;

            Optim.SGD opt = new Optim.SGD(layer.parameters(), 0.01f, 0.9f);

            for (int i = 0; i < 200; i++) {
                opt.zero_grad();
                Tensor loss = Torch.sumTensor(Torch.mul(w, w));
                loss.backward();
                opt.step();
            }
            check(Math.abs(w.data[0]) < 0.1f, "SGD+momentum w[0] near 0, got " + w.data[0]);
            check(Math.abs(w.data[1]) < 0.1f, "SGD+momentum w[1] near 0, got " + w.data[1]);
            System.out.println("  SGD (momentum=0.9) OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // Minimize with Adam
    private static boolean testAdam() {
        try {
            NN.Linear layer = new NN.Linear(new NN(), 2, 1, false);
            Tensor w = layer.weight.getTensor();
            w.data[0] = 5f;
            w.data[1] = -3f;
            w.requires_grad = true;

            Optim.Adam opt = new Optim.Adam(layer.parameters(), 0.1f);

            for (int i = 0; i < 200; i++) {
                opt.zero_grad();
                Tensor loss = Torch.sumTensor(Torch.mul(w, w));
                loss.backward();
                opt.step();
            }
            check(Math.abs(w.data[0]) < 0.1f, "Adam w[0] near 0, got " + w.data[0]);
            check(Math.abs(w.data[1]) < 0.1f, "Adam w[1] near 0, got " + w.data[1]);
            System.out.println("  Adam OK");
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
