package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

import java.util.List;

public class TestBatch4 {

    public static void main(String[] args) {
        testConv1d();
        testBilinear();
        testGroupNorm();
        testOneHot();
        testGRU();
        System.out.println("TEST BATCH 4 PASSED");
    }

    private static void assertEquals(float expected, float actual, float delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new RuntimeException("Expected " + expected + " but got " + actual);
        }
    }

    private static void assertEquals(int expected, int actual) {
        if (expected != actual) {
            throw new RuntimeException("Expected " + expected + " but got " + actual);
        }
    }

    private static void assertNotNull(Object o) {
        if (o == null) {
            throw new RuntimeException("Expected non-null object");
        }
    }

    public static void testConv1d() {
        NN model = new NN();
        NN.Conv1d conv = new NN.Conv1d(model, 2, 2, 3, 1, 0, true);
        
        Tensor x = new Tensor(new float[] { 1,2,3,4,5, 6,7,8,9,10 }, 1, 2, 5);
        x.requires_grad = true;
        
        Tensor out = conv.forward(x);
        assertEquals(1, out.shape[0]);
        assertEquals(2, out.shape[1]);
        assertEquals(3, out.shape[2]);
        
        out.backward();
        assertNotNull(x.grad);
        assertNotNull(conv.weight.getGrad());
        assertNotNull(conv.bias.getGrad());
    }

    public static void testBilinear() {
        NN model = new NN();
        NN.Bilinear bl = new NN.Bilinear(model, 2, 3, 4, true);
        
        Tensor x1 = new Tensor(new float[] { 1, 2 }, 1, 2);
        Tensor x2 = new Tensor(new float[] { 3, 4, 5 }, 1, 3);
        x1.requires_grad = true;
        x2.requires_grad = true;
        
        Tensor out = bl.forward(x1, x2);
        assertEquals(1, out.shape[0]);
        assertEquals(4, out.shape[1]);
        
        out.backward();
        assertNotNull(x1.grad);
        assertNotNull(x2.grad);
        assertNotNull(bl.weight.getGrad());
    }

    public static void testGroupNorm() {
        NN model = new NN();
        NN.GroupNorm gn = new NN.GroupNorm(model, 2, 4);
        
        Tensor x = new Tensor(new float[] { 1,2, 3,4, 5,6, 7,8 }, 1, 4, 2);
        x.requires_grad = true;
        
        Tensor out = gn.forward(x);
        assertEquals(8, out.numel());
        
        out.backward();
        assertNotNull(x.grad);
        assertNotNull(gn.weight.getGrad());
        assertNotNull(gn.bias.getGrad());
    }

    public static void testOneHot() {
        Tensor indices = new Tensor(new float[] { 0, 2, 1 }, 3);
        Tensor oh = Torch.one_hot(indices, 3);
        assertEquals(3, oh.shape[0]);
        assertEquals(3, oh.shape[1]);
        assertEquals(1.0f, oh.data[0 * 3 + 0], 1e-5f);
        assertEquals(1.0f, oh.data[1 * 3 + 2], 1e-5f);
        assertEquals(1.0f, oh.data[2 * 3 + 1], 1e-5f);
        assertEquals(0.0f, oh.data[0 * 3 + 1], 1e-5f);
    }

    public static void testGRU() {
        NN model = new NN();
        NN.GRU gru = new NN.GRU(model, 5, 10, true, true);
        
        Tensor x = new Tensor(1, 3, 5);
        for (int i = 0; i < x.data.length; i++) x.data[i] = (float)Math.sin(i);
        x.requires_grad = true;
        
        Tensor out = gru.forward(x);
        assertEquals(1, out.shape[0]);
        assertEquals(3, out.shape[1]);
        assertEquals(10, out.shape[2]);
        
        out.backward();
        assertNotNull(x.grad);
    }
}
