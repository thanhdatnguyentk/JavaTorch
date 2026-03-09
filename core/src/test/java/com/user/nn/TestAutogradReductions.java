package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestAutogradReductions {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradReductions...");

        boolean allPassed = true;
        allPassed &= testSumTensor();
        allPassed &= testMeanTensor();

        if (allPassed) {
            System.out.println("TestAutogradReductions PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestAutogradReductions FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean condition, String msg) {
        if (!condition) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testSumTensor() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
            a.requires_grad = true;
            Tensor sumA = Torch.sumTensor(a);
            sumA.backward();

            check(a.grad != null, "grad should not be null");
            check(a.grad.dim() == 2, "grad dim check");
            for (int i = 0; i < 4; i++) {
                check(a.grad.data[i] == 1f, "sum elements grad should be 1f");
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testMeanTensor() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
            a.requires_grad = true;
            Tensor meanA = Torch.meanTensor(a);
            meanA.backward();

            check(a.grad != null, "grad should not be null");
            check(a.grad.dim() == 2, "grad dim check");
            for (int i = 0; i < 4; i++) {
                check(Math.abs(a.grad.data[i] - 0.25f) < 1e-6, "mean elements grad should be 1/N (1/4=0.25)");
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
