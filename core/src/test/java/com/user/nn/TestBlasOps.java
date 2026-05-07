package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.core.BlasOps;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestBlasOps {

    @Test
    void testSmallMatmul() {
        // 2x3 * 3x2 = 2x2
        Tensor a = new Tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor b = new Tensor(new float[]{7, 8, 9, 10, 11, 12}, 3, 2);
        Tensor c = Torch.matmul(a, b);
        
        c.toCPU();
        assertEquals(58f, c.data[0], 0.01f);
        assertEquals(64f, c.data[1], 0.01f);
        assertEquals(139f, c.data[2], 0.01f);
        assertEquals(154f, c.data[3], 0.01f);
    }

    @Test
    void testLargerMatmul() {
        int sz = 32;
        Tensor a = new Tensor(sz, sz);
        Tensor b = new Tensor(sz, sz);
        for (int i = 0; i < sz; i++) {
            a.data[i * sz + i] = 1f;
            b.data[i * sz + i] = 2f;
        }
        Tensor c = Torch.matmul(a, b);
        c.toCPU();
        
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                float expected = (i == j) ? 2f : 0f;
                assertEquals(expected, c.data[i * sz + j], 0.01f, "Mismatch at (" + i + "," + j + ")");
            }
        }
    }

    @Test
    void testNonSquare() {
        Tensor a = new Tensor(4, 8);
        Tensor b = new Tensor(8, 3);
        for (int i = 0; i < a.numel(); i++) a.data[i] = (i % 5) * 0.1f;
        for (int i = 0; i < b.numel(); i++) b.data[i] = (i % 7) * 0.2f;

        Tensor c = Torch.matmul(a, b);
        c.toCPU();

        float expected00 = 0f;
        for (int k = 0; k < 8; k++) expected00 += a.data[k] * b.data[k * 3];
        assertEquals(expected00, c.data[0], 0.01f);
    }

    @Test
    void testAutogradWithBlas() {
        Tensor a = new Tensor(new float[]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}, 4, 4);
        a.requires_grad = true;
        Tensor b = new Tensor(new float[]{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}, 4, 4);
        Tensor c = Torch.matmul(a, b);
        Tensor loss = Torch.sumTensor(c);
        loss.backward();
        a.toCPU();
        
        float expected00 = 1+2+3+4; // 10
        assertEquals(expected00, a.grad.data[0], 0.1f);
    }
}
