package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradSimple {

    @Test
    void testElementwiseMulGrad() {
        Tensor a = Torch.tensor(new float[]{2f}, 1);
        Tensor b = Torch.tensor(new float[]{3f}, 1);
        a.requires_grad = true;
        b.requires_grad = true;
        Tensor c = Torch.mul(a, b);
        Tensor s = Torch.sumTensor(c);
        s.backward();
        
        assertEquals(3f, a.grad.data[0], 1e-6f, "autograd mul a.grad mismatch");
        assertEquals(2f, b.grad.data[0], 1e-6f, "autograd mul b.grad mismatch");
    }

    @Test
    void testMatmulGrad() {
        Tensor A = Torch.tensor(new float[]{1f, 2f}, 1, 2);
        A.requires_grad = true;
        Tensor B = Torch.tensor(new float[]{3f, 4f}, 2, 1);
        B.requires_grad = true;
        Tensor C = Torch.matmul(A, B); // 1x1
        Torch.sumTensor(C).backward();
        
        assertEquals(3f, A.grad.data[0], 1e-6f, "autograd matmul A.grad mismatch at [0]");
        assertEquals(4f, A.grad.data[1], 1e-6f, "autograd matmul A.grad mismatch at [1]");
        assertEquals(1f, B.grad.data[0], 1e-6f, "autograd matmul B.grad mismatch at [0]");
        assertEquals(2f, B.grad.data[1], 1e-6f, "autograd matmul B.grad mismatch at [1]");
    }
}
