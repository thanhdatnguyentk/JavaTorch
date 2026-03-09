package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;

public class TestInPlaceOps {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; }
        else { failed++; System.out.println("FAIL: " + name); }
    }

    public static void main(String[] args) {
        testAddScalar();
        testMulScalar();
        testSubScalar();
        testAddTensor();
        testSubTensor();
        testMulTensor();
        testVersionIncrement();
        testVersionCheckDetectsInPlace();
        testVersionCheckPassesWithoutInPlace();

        System.out.println("TestInPlaceOps: " + passed + " passed, " + failed + " failed.");
    }

    static void testAddScalar() {
        Tensor t = new Tensor(new float[]{1, 2, 3}, 3);
        t.add_(10f);
        check("add_ scalar values", t.data[0] == 11f && t.data[1] == 12f && t.data[2] == 13f);
    }

    static void testMulScalar() {
        Tensor t = new Tensor(new float[]{2, 3, 4}, 3);
        t.mul_(3f);
        check("mul_ scalar values", t.data[0] == 6f && t.data[1] == 9f && t.data[2] == 12f);
    }

    static void testSubScalar() {
        Tensor t = new Tensor(new float[]{10, 20, 30}, 3);
        t.sub_(5f);
        check("sub_ scalar values", t.data[0] == 5f && t.data[1] == 15f && t.data[2] == 25f);
    }

    static void testAddTensor() {
        Tensor a = new Tensor(new float[]{1, 2, 3}, 3);
        Tensor b = new Tensor(new float[]{10, 20, 30}, 3);
        a.add_(b);
        check("add_ tensor values", a.data[0] == 11f && a.data[1] == 22f && a.data[2] == 33f);
        // b unchanged
        check("add_ tensor other unchanged", b.data[0] == 10f && b.data[1] == 20f && b.data[2] == 30f);
    }

    static void testSubTensor() {
        Tensor a = new Tensor(new float[]{10, 20, 30}, 3);
        Tensor b = new Tensor(new float[]{1, 2, 3}, 3);
        a.sub_(b);
        check("sub_ tensor values", a.data[0] == 9f && a.data[1] == 18f && a.data[2] == 27f);
    }

    static void testMulTensor() {
        Tensor a = new Tensor(new float[]{2, 3, 4}, 3);
        Tensor b = new Tensor(new float[]{5, 6, 7}, 3);
        a.mul_(b);
        check("mul_ tensor values", a.data[0] == 10f && a.data[1] == 18f && a.data[2] == 28f);
    }

    static void testVersionIncrement() {
        Tensor t = new Tensor(new float[]{1, 2}, 2);
        check("initial version is 0", t.version() == 0);
        t.add_(1f);
        check("version after add_ scalar", t.version() == 1);
        t.mul_(2f);
        check("version after mul_ scalar", t.version() == 2);
        t.sub_(1f);
        check("version after sub_ scalar", t.version() == 3);
        Tensor other = new Tensor(new float[]{1, 1}, 2);
        t.add_(other);
        check("version after add_ tensor", t.version() == 4);
        t.sub_(other);
        check("version after sub_ tensor", t.version() == 5);
        t.mul_(other);
        check("version after mul_ tensor", t.version() == 6);
        t.set(99f, 0);
        check("version after set", t.version() == 7);
    }

    static void testVersionCheckDetectsInPlace() {
        // Build a computation graph: y = x * 2
        Tensor x = new Tensor(new float[]{3f}, 1);
        x.requires_grad = true;
        Tensor y = Torch.mul(x, 2f);

        // In-place modify x AFTER the forward pass
        x.add_(100f);

        // backward should detect the version mismatch
        boolean caught = false;
        try {
            y.backward();
        } catch (RuntimeException e) {
            if (e.getMessage().contains("modified by an in-place operation")) {
                caught = true;
            }
        }
        check("version check detects in-place modification", caught);
    }

    static void testVersionCheckPassesWithoutInPlace() {
        // Normal forward + backward should work fine
        Tensor x = new Tensor(new float[]{3f}, 1);
        x.requires_grad = true;
        Tensor y = Torch.mul(x, 2f);
        y.backward();
        x.toCPU();
        check("normal backward works (grad=2)", Math.abs(x.grad.data[0] - 2f) < 1e-5f);
    }
}
