package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestAutogradSimple {
    public static void main(String[] args) throws Exception {
        int failures = 0;
        try {
            // elementwise mul grad
            Tensor a = Torch.tensor(new float[]{2f},1);
            Tensor b = Torch.tensor(new float[]{3f},1);
            a.requires_grad = true; b.requires_grad = true;
            Tensor c = Torch.mul(a,b);
            Tensor s = Torch.sumTensor(c);
            s.backward();
            if (Math.abs(a.grad.data[0] - 3f) > 1e-6) { System.err.println("autograd mul a.grad wrong: " + a.grad.data[0]); failures++; }
            if (Math.abs(b.grad.data[0] - 2f) > 1e-6) { System.err.println("autograd mul b.grad wrong: " + b.grad.data[0]); failures++; }

            // matmul grad
            Tensor A = Torch.tensor(new float[]{1f,2f},1,2); A.requires_grad=true;
            Tensor B = Torch.tensor(new float[]{3f,4f},2,1); B.requires_grad=true;
            Tensor C = Torch.matmul(A,B); // 1x1
            Torch.sumTensor(C).backward();
            if (Math.abs(A.grad.data[0] - 3f) > 1e-6 || Math.abs(A.grad.data[1] - 4f) > 1e-6) { System.err.println("autograd matmul A.grad wrong: " + java.util.Arrays.toString(A.grad.data)); failures++; }
            if (Math.abs(B.grad.data[0] - 1f) > 1e-6 || Math.abs(B.grad.data[1] - 2f) > 1e-6) { System.err.println("autograd matmul B.grad wrong: " + java.util.Arrays.toString(B.grad.data)); failures++; }

        } catch (Exception e) { e.printStackTrace(); failures++; }
        if (failures==0) { System.out.println("TEST PASSED: Autograd simple"); System.exit(0); } else { System.err.println("TEST FAILED: Autograd simple failures="+failures); System.exit(2); }
    }
}
