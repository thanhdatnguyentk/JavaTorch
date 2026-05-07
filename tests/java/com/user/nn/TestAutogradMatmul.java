package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradMatmul {

    @Test
    void testMatmul() {
        // A(2x3) * B(3x4) = C(2x4)
        Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        Tensor b = Torch.tensor(new float[] {
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3
        }, 3, 4);
        a.requires_grad = true;
        b.requires_grad = true;

        Tensor c = Torch.matmul(a, b);
        Tensor loss = Torch.sumTensor(c);
        loss.backward();

        assertNotNull(a.grad, "a.grad present");
        assertNotNull(b.grad, "b.grad present");
        
        // dA = outGrad * B^T
        // Since loss = sum(C), outGrad is ones(2,4)
        // dA[0,0] = sum(outGrad[0,:] * B[:,0]) = 1*1 + 1*2 + 1*3 = 6? No, wait.
        // Let's re-verify the manual check in the original file: 
        // B rows sum: b_0=4, b_1=8, b_2=12. Correct.
        assertEquals(4f, a.grad.data[0], 1e-6f, "dA[0]");
        assertEquals(8f, a.grad.data[1], 1e-6f, "dA[1]");
        assertEquals(12f, a.grad.data[2], 1e-6f, "dA[2]");
        
        // A cols sum: a_col0(1+4)=5, a_col1(2+5)=7, a_col2(3+6)=9.
        assertEquals(5f, b.grad.data[0], 1e-6f, "dB[0]");
        assertEquals(7f, b.grad.data[4], 1e-6f, "dB[4]");
    }

    @Test
    void testBmm() {
        // A(2x1x2) batch matmul B(2x2x1) = C(2x1x1)
        Tensor a = Torch.tensor(new float[] {
                1, 2, // batch 0
                3, 4 // batch 1
        }, 2, 1, 2);
        Tensor b = Torch.tensor(new float[] {
                1, // batch 0, row 0
                2, // batch 0, row 1

                0, // batch 1, row 0
                1 // batch 1, row 1
        }, 2, 2, 1);

        a.requires_grad = true;
        b.requires_grad = true;

        Tensor c = Torch.bmm(a, b);
        Tensor loss = Torch.sumTensor(c);
        loss.backward();

        assertNotNull(a.grad, "a.grad present");
        assertNotNull(b.grad, "b.grad present");
        
        // dA batch 0: [1, 2], dA batch 1: [0, 1]
        assertEquals(1f, a.grad.data[0], 1e-6f);
        assertEquals(2f, a.grad.data[1], 1e-6f);
        assertEquals(0f, a.grad.data[2], 1e-6f);
        assertEquals(1f, a.grad.data[3], 1e-6f);

        // dB batch 0: [1, 2], dB batch 1: [3, 4]
        assertEquals(1f, b.grad.data[0], 1e-6f);
        assertEquals(2f, b.grad.data[1], 1e-6f);
        assertEquals(3f, b.grad.data[2], 1e-6f);
        assertEquals(4f, b.grad.data[3], 1e-6f);
    }
}
