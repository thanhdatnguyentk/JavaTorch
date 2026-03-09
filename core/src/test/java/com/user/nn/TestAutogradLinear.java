package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.layers.*;

public class TestAutogradLinear {
    public static void main(String[] args) throws Exception {
        int failures = 0;
        try {
            Linear lin = new Linear(3, 2, true);
            // create input Tensor (batch 2 x 3)
            Tensor inp = Torch.tensor(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
            inp.requires_grad = false;
            Tensor out = lin.forward(inp);
            // loss = sum(out)
            Tensor loss = Torch.sumTensor(out);
            loss.backward();
            Tensor w = lin.weight.getTensor();
            Tensor b = lin.bias.getTensor();
            if (w.grad == null) {
                System.err.println("weight.grad null");
                failures++;
            }
            if (b.grad == null) {
                System.err.println("bias.grad null");
                failures++;
            }
            // expected bias grad: ones summed over batch -> 2 for each out feature
            for (int i = 0; i < b.grad.data.length; i++)
                if (Math.abs(b.grad.data[i] - 2f) > 1e-6) {
                    System.err.println("bias.grad incorrect: " + java.util.Arrays.toString(b.grad.data));
                    failures++;
                    break;
                }

        } catch (Exception e) {
            e.printStackTrace();
            failures++;
        }
        if (failures == 0) {
            System.out.println("TEST PASSED: Autograd Linear");
            System.exit(0);
        } else {
            System.err.println("TEST FAILED: Autograd Linear failures=" + failures);
            System.exit(2);
        }
    }
}
