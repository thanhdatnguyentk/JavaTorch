package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.activations.*;

public class TestActivations {
    public static void main(String[] args) {
        NN.Mat m = NN.mat_alloc(3,3);
        // fill with a range
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = (i - 4) * 0.5f; // values -2,-1.5,...

        Sigmoid s = new Sigmoid();
        NN.Mat outS = s.forward(m);
        for (int i = 0; i < outS.es.length; i++) {
            float v = outS.es[i];
            float expected = (float)(1.0 / (1.0 + Math.exp(-m.es[i])));
            if (Math.abs(v - expected) > 1e-6f) { System.err.println("Sigmoid mismatch"); System.exit(1); }
        }

        Tanh t = new Tanh();
        NN.Mat outT = t.forward(m);
        for (int i = 0; i < outT.es.length; i++) {
            float v = outT.es[i];
            float expected = (float)Math.tanh(m.es[i]);
            if (Math.abs(v - expected) > 1e-6f) { System.err.println("Tanh mismatch"); System.exit(2); }
        }

        LeakyReLU lr = new LeakyReLU(0.1f);
        NN.Mat outL = lr.forward(m);
        for (int i = 0; i < outL.es.length; i++) {
            float v = outL.es[i];
            float expected = m.es[i] > 0 ? m.es[i] : 0.1f * m.es[i];
            if (Math.abs(v - expected) > 1e-6f) { System.err.println("LeakyReLU mismatch"); System.exit(3); }
        }

        Softplus sp = new Softplus();
        NN.Mat outP = sp.forward(m);
        for (int i = 0; i < outP.es.length; i++) {
            float v = outP.es[i];
            float expected = (float)Math.log(1.0 + Math.exp(m.es[i]));
            if (Math.abs(v - expected) > 1e-5f) { System.err.println("Softplus mismatch"); System.exit(4); }
        }

        System.out.println("TEST PASSED: Activations");
    }
}
