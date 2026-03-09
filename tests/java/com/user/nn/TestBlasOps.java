package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.core.BlasOps;

public class TestBlasOps {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; }
        else { failed++; System.out.println("FAIL: " + name); }
    }

    public static void main(String[] args) {
        System.out.println("BlasOps available: " + BlasOps.isAvailable());

        testSmallMatmul();
        testLargerMatmul();
        testNonSquare();
        testAutogradWithBlas();
        if (BlasOps.isAvailable()) {
            benchmarkComparison();
        }

        System.out.println("TestBlasOps: " + passed + " passed, " + failed + " failed.");
    }

    static void testSmallMatmul() {
        // 2x3 * 3x2 = 2x2
        Tensor a = new Tensor(new float[]{1,2,3, 4,5,6}, 2, 3);
        Tensor b = new Tensor(new float[]{7,8, 9,10, 11,12}, 3, 2);
        Tensor c = Torch.matmul(a, b);
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[58, 64], [139, 154]]
        c.toCPU();
        check("small matmul [0,0]=58", Math.abs(c.data[0] - 58f) < 0.01f);
        check("small matmul [0,1]=64", Math.abs(c.data[1] - 64f) < 0.01f);
        check("small matmul [1,0]=139", Math.abs(c.data[2] - 139f) < 0.01f);
        check("small matmul [1,1]=154", Math.abs(c.data[3] - 154f) < 0.01f);
    }

    static void testLargerMatmul() {
        // This should use OpenBLAS (m*n*k = 32*32*32 = 32768 > 4096)
        int sz = 32;
        Tensor a = new Tensor(sz, sz);
        Tensor b = new Tensor(sz, sz);
        // Fill with identity-like pattern
        for (int i = 0; i < sz; i++) {
            a.data[i * sz + i] = 1f;
            b.data[i * sz + i] = 2f;
        }
        Tensor c = Torch.matmul(a, b);
        c.toCPU();
        // A=I, B=2I => C=2I
        boolean ok = true;
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                float expected = (i == j) ? 2f : 0f;
                if (Math.abs(c.data[i * sz + j] - expected) > 0.01f) {
                    ok = false;
                    break;
                }
            }
        }
        check("larger matmul (32x32 identity)", ok);
    }

    static void testNonSquare() {
        // 4x8 * 8x3 -> 4x3 (m*n*k = 4*3*8 = 96 < 4096, uses SIMD)
        Tensor a = new Tensor(4, 8);
        Tensor b = new Tensor(8, 3);
        for (int i = 0; i < a.numel(); i++) a.data[i] = (i % 5) * 0.1f;
        for (int i = 0; i < b.numel(); i++) b.data[i] = (i % 7) * 0.2f;

        Tensor c = Torch.matmul(a, b);
        c.toCPU();

        // Verify by manual dot product
        float expected00 = 0f;
        for (int k = 0; k < 8; k++) expected00 += a.data[k] * b.data[k * 3];
        check("non-square matmul [0,0]", Math.abs(c.data[0] - expected00) < 0.01f);
    }

    static void testAutogradWithBlas() {
        // 4x4 * 4x4 with grad
        Tensor a = new Tensor(new float[]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}, 4, 4);
        a.requires_grad = true;
        Tensor b = new Tensor(new float[]{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}, 4, 4);
        Tensor c = Torch.matmul(a, b);
        Tensor loss = Torch.sum_tensor(c);
        loss.backward();
        a.toCPU();
        // d loss/dA = ones(4,4) @ B^T
        // Row i of grad_a = sum over j of B^T[j][col] = sum of B column col
        // grad_a[0][0] = sum of B col 0 = 1+5+9+13 = 28
        // But wait: ones(4,4)@B^T: row i col j = sum_k 1*B^T[k][j] = sum_k B[j][k] = sum of row j of B
        // So grad_a[i][j] = sum of row j of B
        // grad_a[0][0] = sum of B row 0 = 1+2+3+4 = 10
        float expected00 = 1+2+3+4; // = 10
        check("autograd with BLAS: grad ok", Math.abs(a.grad.data[0] - expected00) < 0.1f);
    }

    static void benchmarkComparison() {
        int sz = 256;
        Tensor a = new Tensor(sz, sz);
        Tensor b = new Tensor(sz, sz);
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < a.numel(); i++) a.data[i] = rng.nextFloat();
        for (int i = 0; i < b.numel(); i++) b.data[i] = rng.nextFloat();

        // Warmup
        for (int w = 0; w < 3; w++) Torch.matmul(a, b);

        int iters = 10;
        long start = System.nanoTime();
        for (int i = 0; i < iters; i++) Torch.matmul(a, b);
        long elapsed = System.nanoTime() - start;
        double ms = elapsed / 1e6 / iters;
        System.out.printf("  Matmul %dx%d avg: %.2f ms (OpenBLAS: %s)%n", sz, sz, ms, BlasOps.isAvailable());
        check("benchmark completed", true);
    }
}
