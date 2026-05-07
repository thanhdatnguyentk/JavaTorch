package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.optim.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;

public class TestOptimizers {

    @Test
    void testSGDNoMomentum() {
        Linear layer = new Linear(2, 1, false); // no bias
        Tensor w = layer.weight.getTensor();
        w.data[0] = 5f;
        w.data[1] = -3f;
        w.markDirtyOnCPU();
        w.requires_grad = true;

        Optim.SGD opt = new Optim.SGD(layer.parameters(), 0.1f);

        for (int i = 0; i < 100; i++) {
            opt.zero_grad();
            // f = w[0]^2 + w[1]^2
            Tensor loss = Torch.sumTensor(Torch.mul(w, w));
            loss.backward();
            opt.step();
        }
        
        // After 100 steps, should converge near 0
        assertTrue(Math.abs(w.data[0]) < 0.01f, "SGD w[0] near 0, got " + w.data[0]);
        assertTrue(Math.abs(w.data[1]) < 0.01f, "SGD w[1] near 0, got " + w.data[1]);
    }

    @Test
    void testSGDWithMomentum() {
        Linear layer = new Linear(2, 1, false);
        Tensor w = layer.weight.getTensor();
        w.data[0] = 5f;
        w.data[1] = -3f;
        w.markDirtyOnCPU();
        w.requires_grad = true;

        Optim.SGD opt = new Optim.SGD(layer.parameters(), 0.01f, 0.9f);

        for (int i = 0; i < 200; i++) {
            opt.zero_grad();
            Tensor loss = Torch.sumTensor(Torch.mul(w, w));
            loss.backward();
            opt.step();
        }
        assertTrue(Math.abs(w.data[0]) < 0.1f, "SGD+momentum w[0] near 0, got " + w.data[0]);
        assertTrue(Math.abs(w.data[1]) < 0.1f, "SGD+momentum w[1] near 0, got " + w.data[1]);
    }

    @Test
    void testAdam() {
        Linear layer = new Linear(2, 1, false);
        Tensor w = layer.weight.getTensor();
        w.data[0] = 5f;
        w.data[1] = -3f;
        w.markDirtyOnCPU();
        w.requires_grad = true;

        Optim.Adam opt = new Optim.Adam(layer.parameters(), 0.1f);

        for (int i = 0; i < 200; i++) {
            opt.zero_grad();
            Tensor loss = Torch.sumTensor(Torch.mul(w, w));
            loss.backward();
            opt.step();
        }
        assertTrue(Math.abs(w.data[0]) < 0.1f, "Adam w[0] near 0, got " + w.data[0]);
        assertTrue(Math.abs(w.data[1]) < 0.1f, "Adam w[1] near 0, got " + w.data[1]);
    }
}
