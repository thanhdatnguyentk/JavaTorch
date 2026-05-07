package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.activations.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestActivations {

    @Test
    void testSigmoid() {
        NN.Mat m = NN.mat_alloc(3, 3);
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = (i - 4) * 0.5f;

        Sigmoid s = new Sigmoid();
        NN.Mat out = s.forward(m);
        for (int i = 0; i < out.es.length; i++) {
            float expected = (float) (1.0 / (1.0 + Math.exp(-m.es[i])));
            assertEquals(expected, out.es[i], 1e-6f, "Sigmoid mismatch at index " + i);
        }
    }

    @Test
    void testTanh() {
        NN.Mat m = NN.mat_alloc(3, 3);
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = (i - 4) * 0.5f;

        Tanh t = new Tanh();
        NN.Mat out = t.forward(m);
        for (int i = 0; i < out.es.length; i++) {
            float expected = (float) Math.tanh(m.es[i]);
            assertEquals(expected, out.es[i], 1e-6f, "Tanh mismatch at index " + i);
        }
    }

    @Test
    void testLeakyReLU() {
        NN.Mat m = NN.mat_alloc(3, 3);
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = (i - 4) * 0.5f;

        LeakyReLU lr = new LeakyReLU(0.1f);
        NN.Mat out = lr.forward(m);
        for (int i = 0; i < out.es.length; i++) {
            float expected = m.es[i] > 0 ? m.es[i] : 0.1f * m.es[i];
            assertEquals(expected, out.es[i], 1e-6f, "LeakyReLU mismatch at index " + i);
        }
    }

    @Test
    void testSoftplus() {
        NN.Mat m = NN.mat_alloc(3, 3);
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = (i - 4) * 0.5f;

        Softplus sp = new Softplus();
        NN.Mat out = sp.forward(m);
        for (int i = 0; i < out.es.length; i++) {
            float expected = (float) Math.log(1.0 + Math.exp(m.es[i]));
            assertEquals(expected, out.es[i], 1e-5f, "Softplus mismatch at index " + i);
        }
    }
}
