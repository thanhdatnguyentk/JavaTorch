package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradLinear {

    @Test
    void testLinearAutograd() {
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
        
        assertNotNull(w.grad, "Weight gradient should not be null");
        assertNotNull(b.grad, "Bias gradient should not be null");
        
        // expected bias grad: ones summed over batch -> 2 for each out feature
        for (int i = 0; i < b.grad.data.length; i++) {
            assertEquals(2f, b.grad.data[i], 1e-6, "Bias gradient incorrect at index " + i);
        }
    }

    @Test
    void testLinearAutogradWithInputGrad() {
        Linear lin = new Linear(3, 2, false); // no bias
        Tensor inp = Torch.ones(1, 3);
        inp.requires_grad = true;
        
        // Set weights to ones
        Tensor w = lin.weight.getTensor();
        for(int i=0; i<w.data.length; i++) w.data[i] = 1.0f;
        w.markDirtyOnCPU();
        
        Tensor out = lin.forward(inp);
        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        
        // out = inp * W^T
        // loss = sum(out)
        // dloss/dinp = sum(dloss/dout * dout/dinp) = ones(1,2) * W = [2, 2, 2]
        assertNotNull(inp.grad, "Input gradient should not be null");
        for(int i=0; i<inp.grad.data.length; i++) {
            assertEquals(2.0f, inp.grad.data[i], 1e-6, "Input gradient incorrect at index " + i);
        }
    }
}
