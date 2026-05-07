package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestEdgeCases {

    @Test
    void testUnsqueezeBoundsCheck() {
        Tensor t = Torch.tensor(new float[]{1, 2, 3}, 3); // 1D, shape [3]

        // Valid dims: 0 and 1 for a 1D tensor
        Tensor u0 = t.unsqueeze(0);
        assertArrayEquals(new int[]{1, 3}, u0.shape, "unsqueeze(0) shape mismatch");

        Tensor u1 = t.unsqueeze(1);
        assertArrayEquals(new int[]{3, 1}, u1.shape, "unsqueeze(1) shape mismatch");

        // Negative indexing: -1 should equal dim=1 for 1D
        Tensor uNeg = t.unsqueeze(-1);
        assertArrayEquals(new int[]{3, 1}, uNeg.shape, "unsqueeze(-1) shape mismatch");

        // Out-of-bounds: dim=5 on 1D tensor should throw
        assertThrows(IndexOutOfBoundsException.class, () -> t.unsqueeze(5));

        // Out-of-bounds: dim=-10 on 1D tensor should throw
        assertThrows(IndexOutOfBoundsException.class, () -> t.unsqueeze(-10));
    }

    @Test
    void testSqueezeNoOp() {
        // squeeze on a tensor with no size-1 dims should be a no-op
        Tensor t = Torch.tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor s = t.squeeze();
        assertEquals(2, s.dim(), "squeeze on [2,3] should stay 2D");
        assertArrayEquals(new int[]{2, 3}, s.shape, "squeeze on [2,3] should preserve shape");

        // squeeze on 1D tensor
        Tensor t1d = Torch.tensor(new float[]{1, 2, 3}, 3);
        Tensor s1d = t1d.squeeze();
        assertEquals(1, s1d.dim());
        assertEquals(3, s1d.numel());
    }

    @Test
    void testDetach() {
        Tensor a = Torch.tensor(new float[]{2f, 3f}, 2);
        a.requires_grad = true;
        Tensor b = Torch.mul(a, a); // b = a^2, has grad_fn

        Tensor d = b.detach();
        assertFalse(d.requires_grad, "detach() should set requires_grad=false");
        assertNull(d.grad_fn, "detach() should clear grad_fn");
        assertEquals(4f, d.data[0], 1e-6f);
        assertEquals(9f, d.data[1], 1e-6f);
    }

    @Test
    void testZeroGrad() {
        Tensor t = Torch.tensor(new float[]{1f, 2f, 3f}, 3);
        t.requires_grad = true;
        Tensor loss = Torch.sumTensor(Torch.mul(t, t)); // sum(t^2)
        loss.backward();
        assertNotNull(t.grad, "grad should exist after backward");
        t.zero_grad();
        assertNull(t.grad, "zero_grad() should clear grad");
    }

    @Test
    void testGradAccumulation() {
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
        assertEquals(g0_first * 2, t.grad.data[0], 1e-4f, "Grad accumulation failure");
        assertEquals(g1_first * 2, t.grad.data[1], 1e-4f, "Grad accumulation failure");
    }

    @Test
    void testSingleElementOps() {
        Tensor scalar = Torch.tensor(new float[]{5f}, 1);
        assertEquals(1, scalar.numel());
        assertEquals(1, scalar.dim());

        Tensor reshaped = scalar.reshape(1, 1);
        assertEquals(2, reshaped.dim(), "reshape scalar to [1,1]");
        assertArrayEquals(new int[]{1, 1}, reshaped.shape);

        Tensor sum = Torch.sumTensor(scalar);
        assertEquals(5f, sum.data[0], 1e-6f, "sum of single element");
    }
}
