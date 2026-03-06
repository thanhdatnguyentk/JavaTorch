package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestFunctional {
    public static void main(String[] args) {
        NN lib = new NN();
        NN.Mat m = lib.mat_alloc(2,2);
        lib.mat_fill(m, -0.5f);
        m.es[0] = 1.0f; // one positive

        NN.Mat r1 = NN.F.relu(lib, m);
        NN.ReLU relu = new NN.ReLU();
        NN.Mat r2 = relu.forward(m);

        for (int i = 0; i < r1.es.length; i++) {
            if (Math.abs(r1.es[i] - r2.es[i]) > 1e-6f) {
                System.err.println("F.relu mismatch vs ReLU");
                System.exit(1);
            }
        }
        System.out.println("TEST PASSED: Functional");
    }
}
