package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import java.io.File;
import java.io.IOException;
import static org.junit.jupiter.api.Assertions.*;

public class TestTorchCoverage {

    @Test
    void testCreationOps() {
        Tensor z = Torch.zeros(2, 3);
        Tensor o = Torch.ones(2, 3);
        Tensor r = Torch.arange(0, 6).reshape(2, 3);
        Tensor l = Torch.linspace(0f, 1f, 5);
        Tensor id = Torch.eye(3);
        
        assertEquals(6, z.numel());
        assertEquals(6, o.numel());
        assertEquals(6, r.numel());
        assertEquals(5, l.numel());
        assertEquals(9, id.numel());
        assertEquals(1.0f, id.get(0, 0));
        assertEquals(0.0f, id.get(0, 1));
    }

    @Test
    void testBasicMath() {
        Tensor r = Torch.arange(0, 6).reshape(2, 3);
        Tensor s = Torch.add(r, 1.0f);
        Tensor p = Torch.mul(s, 2.0f);
        assertEquals((r.data[0] + 1f) * 2f, p.data[0], 1e-6f);
    }

    @Test
    void testBroadcast() {
        Tensor a = new Tensor(new float[]{1, 2, 3}, 3);
        Tensor b = Torch.full(new int[]{3, 3}, 2f);
        // shape [3,1] + [3,3] -> [3,3]
        Tensor c = Torch.add(a.reshape(3, 1), b); 
        assertEquals(3, c.shape[0]);
        assertEquals(3, c.shape[1]);
        assertEquals(3.0f, c.get(0, 0), 1e-6f); // 1 + 2
        assertEquals(5.0f, c.get(2, 0), 1e-6f); // 3 + 2
    }

    @Test
    void testReductions() {
        Tensor a = Torch.ones(2, 2);
        float sum = Torch.sum(a);
        assertEquals(4.0f, sum, 1e-6f);
        float mean = Torch.mean(a);
        assertEquals(1.0f, mean, 1e-6f);
    }

    @Test
    void testMatmul() {
        Tensor A = Torch.tensor(new float[]{1, 2, 3, 4}, 2, 2);
        Tensor B = Torch.tensor(new float[]{5, 6, 7, 8}, 2, 2);
        Tensor M = Torch.matmul(A, B);
        assertArrayEquals(new int[]{2, 2}, M.shape);
        // [1*5 + 2*7] = 19
        assertEquals(19.0f, M.data[0], 1e-6f);
    }

    @Test
    void testSaveLoad() throws IOException {
        Tensor rnd = Torch.randint(0, 10, 3, 3);
        File tempFile = File.createTempFile("tensor_save", ".txt");
        String path = tempFile.getAbsolutePath();
        
        Torch.save(rnd, path);
        Tensor loaded = Torch.load(path);
        
        assertEquals(rnd.numel(), loaded.numel());
        assertArrayEquals(rnd.shape, loaded.shape);
        assertArrayEquals(rnd.data, loaded.data, 1e-6f);
        
        tempFile.delete();
    }

    @Test
    void testGradControl() throws Exception {
        Torch.enable_grad();
        assertTrue(Torch.is_grad_enabled());
        
        try (AutoCloseable c = Torch.no_grad()) {
            assertFalse(Torch.is_grad_enabled(), "no_grad should disable grad");
        }
        
        assertTrue(Torch.is_grad_enabled(), "grad should be re-enabled after no_grad block");
    }
}
