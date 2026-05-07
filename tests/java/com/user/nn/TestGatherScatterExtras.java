package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestGatherScatterExtras {

    @Test
    void testGatherNegativeIndices() {
        Tensor inp = Torch.tensor(new float[]{10, 11, 12, 20, 21, 22}, 2, 3);
        // idxNeg: -1 refers to index 2, -2 refers to index 1
        Tensor idxNeg = Torch.tensor(new float[]{-1, 0, 1, -2, -1, 0}, 2, 3);
        Tensor g = Torch.gather(inp, 1, idxNeg);
        
        // Expected: 
        // Row 0: inp[0, 2]=12, inp[0, 0]=10, inp[0, 1]=11
        // Row 1: inp[1, 1]=21, inp[1, 2]=22, inp[1, 0]=20
        assertEquals(12f, g.data[0]);
        assertEquals(10f, g.data[1]);
        assertEquals(11f, g.data[2]);
        assertEquals(21f, g.data[3]);
        assertEquals(22f, g.data[4]);
        assertEquals(20f, g.data[5]);
    }

    @Test
    void testScatterInPlace() {
        Tensor base = Torch.zeros(2, 3);
        Tensor index = Torch.tensor(new float[]{0, 1, 2, 0, 1, 2}, 2, 3);
        Tensor src = Torch.tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Torch.scatter_(base, 1, index, src);
        
        assertEquals(1f, base.data[0]);
        assertEquals(2f, base.data[1]);
        assertEquals(3f, base.data[2]);
        assertEquals(4f, base.data[3]);
        assertEquals(5f, base.data[4]);
        assertEquals(6f, base.data[5]);
    }
}
