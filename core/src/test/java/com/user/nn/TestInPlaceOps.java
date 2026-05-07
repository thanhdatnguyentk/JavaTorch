package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestInPlaceOps {

    @Test
    void testAddScalar() {
        Tensor t = new Tensor(new float[]{1, 2, 3}, 3);
        t.add_(10f);
        assertArrayEquals(new float[]{11f, 12f, 13f}, t.data, 1e-6f);
    }

    @Test
    void testMulScalar() {
        Tensor t = new Tensor(new float[]{2, 3, 4}, 3);
        t.mul_(3f);
        assertArrayEquals(new float[]{6f, 9f, 12f}, t.data, 1e-6f);
    }

    @Test
    void testSubScalar() {
        Tensor t = new Tensor(new float[]{10, 20, 30}, 3);
        t.sub_(5f);
        assertArrayEquals(new float[]{5f, 15f, 25f}, t.data, 1e-6f);
    }

    @Test
    void testAddTensor() {
        Tensor a = new Tensor(new float[]{1, 2, 3}, 3);
        Tensor b = new Tensor(new float[]{10, 20, 30}, 3);
        a.add_(b);
        assertArrayEquals(new float[]{11f, 22f, 33f}, a.data, 1e-6f);
        assertArrayEquals(new float[]{10f, 20f, 30f}, b.data, 1e-6f, "b should be unchanged");
    }

    @Test
    void testSubTensor() {
        Tensor a = new Tensor(new float[]{10, 20, 30}, 3);
        Tensor b = new Tensor(new float[]{1, 2, 3}, 3);
        a.sub_(b);
        assertArrayEquals(new float[]{9f, 18f, 27f}, a.data, 1e-6f);
    }

    @Test
    void testMulTensor() {
        Tensor a = new Tensor(new float[]{2, 3, 4}, 3);
        Tensor b = new Tensor(new float[]{5, 6, 7}, 3);
        a.mul_(b);
        assertArrayEquals(new float[]{10f, 18f, 28f}, a.data, 1e-6f);
    }

    @Test
    void testVersionIncrement() {
        Tensor t = new Tensor(new float[]{1, 2}, 2);
        assertEquals(0, t.version(), "initial version is 0");
        t.add_(1f);
        assertEquals(1, t.version());
        t.mul_(2f);
        assertEquals(2, t.version());
        t.sub_(1f);
        assertEquals(3, t.version());
        
        Tensor other = new Tensor(new float[]{1, 1}, 2);
        t.add_(other);
        assertEquals(4, t.version());
        t.sub_(other);
        assertEquals(5, t.version());
        t.mul_(other);
        assertEquals(6, t.version());
        
        t.set(99f, 0);
        assertEquals(7, t.version());
    }

    @Test
    void testVersionCheckDetectsInPlace() {
        // Build a computation graph: y = x * 2
        Tensor x = new Tensor(new float[]{3f}, 1);
        x.requires_grad = true;
        Tensor y = Torch.mul(x, 2f);

        // In-place modify x AFTER the forward pass
        x.add_(100f);

        // backward should detect the version mismatch
        RuntimeException ex = assertThrows(RuntimeException.class, y::backward);
        assertTrue(ex.getMessage().contains("modified by an in-place operation"));
    }

    @Test
    void testVersionCheckPassesWithoutInPlace() {
        // Normal forward + backward should work fine
        Tensor x = new Tensor(new float[]{3f}, 1);
        x.requires_grad = true;
        Tensor y = Torch.mul(x, 2f);
        y.backward();
        x.toCPU();
        assertEquals(2.0f, x.grad.data[0], 1e-5f, "normal backward works (grad=2)");
    }
}
