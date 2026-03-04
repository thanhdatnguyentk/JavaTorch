package com.user.nn;

public class TestMatOps {
    public static void main(String[] args) {
        nn lib = new nn();
        nn.Mat a = lib.mat_alloc(2, 3);
        nn.Mat b = lib.mat_alloc(3, 2);
        // fill a and b with deterministic values
        lib.mat_rand_seed(a, 1L, -1f, 1f);
        lib.mat_rand_seed(b, 2L, -1f, 1f);

        nn.Mat out = lib.mat_alloc(2, 2);
        lib.mat_dot(out, a, b);
        // basic sanity: dimensions
        if (out.rows != 2 || out.cols != 2) {
            System.err.println("mat_dot produced wrong shape");
            System.exit(1);
        }

        // test sum and sub
        nn.Mat copy = lib.mat_alloc(out.rows, out.cols);
        System.arraycopy(out.es, 0, copy.es, 0, out.es.length);
        lib.mat_sum(copy, out); // copy = copy + out -> equals out*2
        for (int i = 0; i < copy.es.length; i++) {
            if (Math.abs(copy.es[i] - 2f * out.es[i]) > 1e-6f) {
                System.err.println("mat_sum mismatch");
                System.exit(2);
            }
        }

        lib.mat_sub(copy, out); // copy = copy - out -> back to out
        for (int i = 0; i < copy.es.length; i++) {
            if (Math.abs(copy.es[i] - out.es[i]) > 1e-6f) {
                System.err.println("mat_sub mismatch");
                System.exit(3);
            }
        }

        // test apply inplace
        lib.mat_apply_inplace(out, new nn.ElemOp(){ public float apply(float x){ return x*0f; }});
        for (float v : out.es) if (Math.abs(v) > 1e-6f) { System.err.println("mat_apply_inplace failed"); System.exit(4); }

        System.out.println("TEST PASSED: MatOps");
    }
}
