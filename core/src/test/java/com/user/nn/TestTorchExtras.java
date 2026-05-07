package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class TestTorchExtras {

    @Test
    void testStackCatSplitChunk() {
        Tensor t1 = Torch.tensor(new float[]{1, 2}, 2);
        Tensor t2 = Torch.tensor(new float[]{3, 4}, 2);
        List<Tensor> lst = new ArrayList<>(); 
        lst.add(t1); lst.add(t2);
        
        Tensor stacked = Torch.stack(lst, 0); // shape (2,2)
        assertArrayEquals(new int[]{2, 2}, stacked.shape, "stack shape wrong");
        assertEquals(1f, stacked.data[0]);
        assertEquals(3f, stacked.data[2]);

        Tensor seq = Torch.arange(0, 6).reshape(6);
        List<Tensor> parts = Torch.split(seq, new int[]{2, 2, 2}, 0);
        assertEquals(3, parts.size(), "split size wrong");
        assertEquals(2, parts.get(0).numel(), "split partition size wrong");

        List<Tensor> chunks = Torch.chunk(seq, 3, 0);
        assertEquals(3, chunks.size(), "chunk size wrong");
    }

    @Test
    void testWhere() {
        Tensor cond = Torch.tensor(new float[]{1, 0, 1, 0}, 4);
        Tensor xa = Torch.tensor(new float[]{10, 11, 12, 13}, 4);
        Tensor ya = Torch.tensor(new float[]{100, 101, 102, 103}, 4);
        Tensor w = Torch.where(cond, xa, ya);
        assertEquals(10f, w.data[0]);
        assertEquals(101f, w.data[1]);
        assertEquals(12f, w.data[2]);
        assertEquals(103f, w.data[3]);
    }

    @Test
    void testPermute() {
        Tensor A = Torch.arange(0, 6).reshape(2, 3);
        Tensor At = Torch.permute(A, 1, 0); // shape (3,2)
        assertArrayEquals(new int[]{3, 2}, At.shape);
        assertEquals(A.data[0], At.data[0]);
        assertEquals(A.get(0, 1), At.get(1, 0));
    }

    @Test
    void testGatherScatter() {
        Tensor inp = Torch.tensor(new float[]{10, 11, 12, 20, 21, 22}, 2, 3);
        Tensor idx = Torch.tensor(new float[]{2, 0, 1, 1, 2, 0}, 2, 3);
        Tensor g = Torch.gather(inp, 1, idx);
        assertEquals(12f, g.data[0]);
        assertEquals(21f, g.data[3]);

        Tensor zeros = Torch.zeros(2, 3);
        Tensor s = Torch.scatter(zeros, 1, idx, Torch.tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3));
        // Resulting layout row 0: at index 2 put 1, at index 0 put 2, at index 1 put 3 -> [2, 3, 1]
        // Row 1: at index 1 put 4, at index 2 put 5, at index 0 put 6 -> [6, 4, 5]
        assertEquals(2f, s.data[0]);
        assertEquals(3f, s.data[1]);
        assertEquals(1f, s.data[2]);
        assertEquals(6f, s.data[3]);
        assertEquals(4f, s.data[4]);
        assertEquals(5f, s.data[5]);
    }

    @Test
    void testMathFunctions() {
        Tensor v = Torch.tensor(new float[]{0.5f, -0.5f}, 2);
        assertNotNull(Torch.asin(v));
        assertNotNull(Torch.acos(v));
        assertNotNull(Torch.atan(v));
        
        Tensor l10 = Torch.log10(Torch.tensor(new float[]{10f}, 1));
        assertEquals(1.0f, l10.data[0], 1e-6f);
        
        Tensor l2 = Torch.log2(Torch.tensor(new float[]{8f}, 1));
        assertEquals(3.0f, l2.data[0], 1e-6f);
        
        Tensor tr = Torch.trunc(Torch.tensor(new float[]{1.9f, -1.9f}, 2));
        assertEquals(1f, tr.data[0]);
        assertEquals(-1f, tr.data[1]);
        
        Tensor cmp = Torch.ge(Torch.tensor(new float[]{1, 2}, 2), Torch.tensor(new float[]{1, 3}, 2));
        assertEquals(1f, cmp.data[0]);
        assertEquals(0f, cmp.data[1]);
    }

    @Test
    void testReductionsExtras() {
        Tensor numbers = Torch.tensor(new float[]{1f, 2f, 3f, 4f}, 4);
        assertEquals(1.25f, Torch.var(numbers), 1e-6f);
        assertEquals((float)Math.sqrt(1.25), Torch.std(numbers), 1e-6f);
        
        Tensor mat2 = Torch.tensor(new float[]{5f, 2f, 3f, 1f, 4f, 6f}, 2, 3);
        int[] amin = Torch.argmin(mat2, 1);
        assertArrayEquals(new int[]{1, 0}, amin);
        
        Tensor v2 = Torch.tensor(new float[]{3f, 4f}, 2);
        assertEquals(5f, Torch.norm(v2), 1e-6f);
    }

    @Test
    void testLinearAlgebra() {
        Tensor M = Torch.tensor(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        assertEquals(-2f, Torch.det(M), 1e-6f);
        
        Tensor Minv = Torch.inverse(M);
        // det = -2. adj = [[4,-2],[-3,1]]. Minv = [[-2,1],[1.5,-0.5]]
        assertEquals(-2f, Minv.data[0], 1e-6f);
        assertEquals(1f, Minv.data[1], 1e-6f);
        assertEquals(1.5f, Minv.data[2], 1e-6f);
        assertEquals(-0.5f, Minv.data[3], 1e-6f);
    }

    @Test
    void testProbabilisticOps() {
        Tensor b0 = Torch.bernoulli(0.0f, 10);
        for (float v : b0.data) assertEquals(0f, v);
        
        Tensor b1 = Torch.bernoulli(1.0f, 5);
        for (float v : b1.data) assertEquals(1f, v);
        
        Tensor probs = Torch.tensor(new float[]{0f, 1f, 0f}, 3);
        Tensor m = Torch.multinomial(probs, 3, true);
        for (float v : m.data) assertEquals(1f, v);
    }
}
