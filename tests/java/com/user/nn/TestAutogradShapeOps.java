package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestAutogradShapeOps {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradShapeOps...");

        boolean allPassed = true;
        allPassed &= testReshape();
        allPassed &= testPermute();
        allPassed &= testTranspose();
        allPassed &= testSqueeze();

        if (allPassed) {
            System.out.println("TestAutogradShapeOps PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestAutogradShapeOps FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean condition, String msg) {
        if (!condition) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testReshape() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
            a.requires_grad = true;
            Tensor b = a.reshape(4);
            // arbitrary op to check backward: use sum which should push grad of 1s
            Tensor loss = Torch.sumTensor(b);
            loss.backward();

            check(a.grad != null, "grad should not be null");
            check(a.grad.dim() == 2, "grad should have original shape");
            check(a.grad.shape[0] == 2 && a.grad.shape[1] == 2, "grad shape should be 2x2");
            for (int i = 0; i < 4; i++)
                check(a.grad.data[i] == 1f, "grad should be 1f");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testPermute() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            a.requires_grad = true;
            Tensor b = Torch.permute(a, 1, 0); // shape: 3, 2

            // To test proper gradient flow:
            // Let's multiply by a matching mask before sum
            Tensor mask = Torch.tensor(new float[] { 1, 0, 0, 1, 0, 0 }, 3, 2);
            Tensor loss = Torch.sumTensor(Torch.mul(b, mask));
            loss.backward();

            // b = [[1, 4], [2, 5], [3, 6]]
            // mask = [[1, 0], [0, 1], [0, 0]]
            // so mask on original elements means a_00 (1) and a_11 (5) receive a grad of 1.

            check(a.grad.data[0] == 1f, "a_00 grad");
            check(a.grad.data[4] == 1f, "a_11 grad");
            check(a.grad.data[1] == 0f, "a_01 grad");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testTranspose() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            a.requires_grad = true;
            Tensor b = Torch.transpose(a, 0, 1);
            Tensor loss = Torch.sumTensor(b);
            loss.backward();

            check(a.grad.shape[0] == 2 && a.grad.shape[1] == 3, "grad shape");
            for (int i = 0; i < 6; i++)
                check(a.grad.data[i] == 1f, "grad should be 1f");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testSqueeze() {
        try {
            Tensor a = Torch.tensor(new float[] { 1, 2 }, 1, 2, 1);
            a.requires_grad = true;
            Tensor b = a.squeeze(); // shape: 2
            Tensor c = b.unsqueeze(1); // shape: 2, 1

            Tensor loss = Torch.sumTensor(c);
            loss.backward();

            check(a.grad.dim() == 3, "grad dim matches a");
            check(a.grad.shape[0] == 1 && a.grad.shape[1] == 2 && a.grad.shape[2] == 1, "grad shape");
            for (int i = 0; i < 2; i++)
                check(a.grad.data[i] == 1f, "grad should be 1f");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
