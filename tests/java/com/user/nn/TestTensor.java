package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestTensor {

    @Test
    void testConstructorAndShape() {
        Tensor t = new Tensor(2, 3);
        assertEquals(6, t.numel(), "Numel should be product of shape dims");
        assertEquals(2, t.dim(), "Dim should match shape length");
        assertArrayEquals(new int[]{2, 3}, t.shape, "Shape should match constructor args");
    }

    @Test
    void testGetSet() {
        Tensor t = new Tensor(2, 3);
        t.set(5.0f, 1, 2);
        assertEquals(5.0f, t.get(1, 2), "Get should return value set by set()");
    }

    @Test
    void testReshape() {
        Tensor r = new Tensor(new float[]{0, 1, 2, 3, 4, 5}, 6).reshape(2, 3);
        assertEquals(2, r.shape[0]);
        assertEquals(3, r.shape[1]);
        assertEquals(5f, r.get(1, 2), "Value at [1,2] should be 5 after reshape");
        
        // Test -1 reshape
        Tensor r2 = r.reshape(-1);
        assertEquals(1, r2.dim());
        assertEquals(6, r2.shape[0]);
    }

    @Test
    void testClone() {
        Tensor r = new Tensor(new float[]{0, 1, 2, 3, 4, 5}, 2, 3);
        Tensor c = r.clone();
        c.set(9f, 0, 0);
        assertNotEquals(r.get(0, 0), c.get(0, 0), "Clone should be deep copy of data");
        assertEquals(0f, r.get(0, 0), "Original data should not be modified by clone modification");
    }

    @Test
    void testFlatten() {
        Tensor r = new Tensor(2, 3);
        Tensor f = r.flatten();
        assertEquals(1, f.dim(), "Flattened tensor should have rank 1");
        assertEquals(6, f.numel(), "Flattened tensor should preserve numel");
    }

    @Test
    void testSqueezeUnsqueeze() {
        Tensor a = new Tensor(new float[]{7f}, 1, 1, 1).squeeze();
        // Squeeze removes all dims of size 1 if they are not the only dim
        // In current implementation, if all are 1, it might return rank 1 with 1 element
        assertTrue(a.dim() <= 1, "Squeeze should reduce rank");
        
        Tensor b = a.unsqueeze(0);
        assertEquals(1, b.shape[0], "Unsqueeze(0) should add dim 1 at start");
    }

    @Test
    void testInplaceOps() {
        Tensor ip = Torch.tensor(new float[]{1f, 2f, 3f}, 3);
        ip.add_(2f);
        assertEquals(3f, ip.data[0], 1e-6, "add_(2f) failed");
        ip.mul_(2f);
        assertEquals(8f, ip.data[1], 1e-6, "mul_(2f) failed");
    }
    
    @Test
    void testDetach() {
        Tensor t = Torch.rand(new int[]{2, 2});
        t.requires_grad = true;
        Tensor d = t.detach();
        assertFalse(d.requires_grad, "Detached tensor should not require grad");
        assertNull(d.grad_fn, "Detached tensor should not have grad_fn");
        assertArrayEquals(t.shape, d.shape, "Detached tensor should have same shape");
    }
}
