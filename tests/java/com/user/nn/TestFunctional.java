package com.user.nn;

public class TestFunctional {
    public static void main(String[] args) {
        nn lib = new nn();
        nn.Mat m = lib.mat_alloc(2,2);
        lib.mat_fill(m, -0.5f);
        m.es[0] = 1.0f; // one positive

        nn.Mat r1 = nn.F.relu(lib, m);
        nn.ReLU relu = new nn.ReLU();
        nn.Mat r2 = relu.forward(m);

        for (int i = 0; i < r1.es.length; i++) {
            if (Math.abs(r1.es[i] - r2.es[i]) > 1e-6f) {
                System.err.println("F.relu mismatch vs ReLU");
                System.exit(1);
            }
        }
        System.out.println("TEST PASSED: Functional");
    }
}
