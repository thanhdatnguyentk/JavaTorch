package com.user.nn;

import com.user.nn.core.*;

/**
 * Tests for edge cases: unsqueeze bounds, squeeze no-op, detach, zero_grad,
 * gradient accumulation, and single-element tensor operations.
 */
public class TestEdgeCases {
    static int failures = 0;

    public static void main(String[] args) {
        System.out.println("Running TestEdgeCases...");

        testUnsqueezeBoundsCheck();
        testSqueezeNoOp();
        testDetach();
        testZeroGrad();
        testGradAccumulation();
        testSingleElementOps();

        if (failures > 0) {
            System.out.println("TestEdgeCases FAILED (" + failures + " failures).");
            System.exit(1);
        }
        System.out.println("TestEdgeCases PASSED.");
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            failures++;
        }
    }

    private static void testUnsqueezeBoundsCheck() {
        Tensor t = Torch.tensor(new float[]{1, 2, 3}, 3); // 1D, shape [3]

        // Valid dims: 0 and 1 for a 1D tensor
        Tensor u0 = t.unsqueeze(0);
        check(u0.shape[0] == 1 && u0.shape[1] == 3, "unsqueeze(0) shape mismatch");

        Tensor u1 = t.unsqueeze(1);
        check(u1.shape[0] == 3 && u1.shape[1] == 1, "unsqueeze(1) shape mismatch");

        // Negative indexing: -1 should equal dim=1 for 1D
        Tensor uNeg = t.unsqueeze(-1);
        check(uNeg.shape[0] == 3 && uNeg.shape[1] == 1, "unsqueeze(-1) shape mismatch");

        // Out-of-bounds: dim=5 on 1D tensor should throw
        boolean threw = false;
        try {
            t.unsqueeze(5);
        } catch (IndexOutOfBoundsException e) {
            threw = true;
        }
        check(threw, "unsqueeze(5) on 1D tensor should throw IndexOutOfBoundsException");

        // Out-of-bounds: dim=-10 on 1D tensor should throw
        threw = false;
        try {
            t.unsqueeze(-10);
        } catch (IndexOutOfBoundsException e) {
            threw = true;
        }
        check(threw, "unsqueeze(-10) on 1D tensor should throw IndexOutOfBoundsException");

        System.out.println("  Unsqueeze bounds check OK");
    }

    private static void testSqueezeNoOp() {
        // squeeze on a tensor with no size-1 dims should be a no-op
        Tensor t = Torch.tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor s = t.squeeze();
        check(s.dim() == 2, "squeeze on [2,3] should stay 2D, got " + s.dim());
        check(s.shape[0] == 2 && s.shape[1] == 3, "squeeze on [2,3] should preserve shape");

        // squeeze on 1D tensor
        Tensor t1d = Torch.tensor(new float[]{1, 2, 3}, 3);
        Tensor s1d = t1d.squeeze();
        check(s1d.dim() == 1 && s1d.numel() == 3, "squeeze on 1D should be no-op");

        System.out.println("  Squeeze no-op OK");
    }

    private static void testDetach() {
        Tensor a = Torch.tensor(new float[]{2f, 3f}, 2);
        a.requires_grad = true;
        Tensor b = Torch.mul(a, a); // b = a^2, has grad_fn

        Tensor d = b.detach();
        check(!d.requires_grad, "detach() should set requires_grad=false");
        check(d.grad_fn == null, "detach() should clear grad_fn");
        // values should be same
        check(Math.abs(d.data[0] - 4f) < 1e-6f, "detach() should preserve values");
        check(Math.abs(d.data[1] - 9f) < 1e-6f, "detach() should preserve values");

        System.out.println("  Detach OK");
    }

    private static void testZeroGrad() {
        Tensor t = Torch.tensor(new float[]{1f, 2f, 3f}, 3);
        t.requires_grad = true;
        Tensor loss = Torch.sumTensor(Torch.mul(t, t)); // sum(t^2)
        loss.backward();
        check(t.grad != null, "grad should exist after backward");
        t.zero_grad();
        check(t.grad == null, "zero_grad() should clear grad");

        System.out.println("  Zero grad OK");
    }

    private static void testGradAccumulation() {
        Tensor t = Torch.tensor(new float[]{1f, 2f}, 2);
        t.requires_grad = true;

        // First backward: grad = 2*t = [2, 4]
        Tensor loss1 = Torch.sumTensor(Torch.mul(t, t));
        loss1.backward();
        float g0_first = t.grad.data[0];
        float g1_first = t.grad.data[1];

        // Second backward without zero_grad: grads should accumulate
        Tensor loss2 = Torch.sumTensor(Torch.mul(t, t));
        loss2.backward();
        check(Math.abs(t.grad.data[0] - g0_first * 2) < 1e-4f,
              "Grad accumulation: expected " + (g0_first * 2) + ", got " + t.grad.data[0]);
        check(Math.abs(t.grad.data[1] - g1_first * 2) < 1e-4f,
              "Grad accumulation: expected " + (g1_first * 2) + ", got " + t.grad.data[1]);

        System.out.println("  Gradient accumulation OK");
    }

    private static void testSingleElementOps() {
        Tensor scalar = Torch.tensor(new float[]{5f}, 1);
        check(scalar.numel() == 1, "single element numel");
        check(scalar.dim() == 1, "single element dim");

        Tensor reshaped = scalar.reshape(1, 1);
        check(reshaped.dim() == 2, "reshape scalar to [1,1]");
        check(reshaped.shape[0] == 1 && reshaped.shape[1] == 1, "reshape scalar shape");

        Tensor sum = Torch.sumTensor(scalar);
        check(Math.abs(sum.data[0] - 5f) < 1e-6f, "sum of single element");

        System.out.println("  Single element ops OK");
    }
}
