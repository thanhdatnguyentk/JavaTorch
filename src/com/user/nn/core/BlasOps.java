package com.user.nn.core;

import static org.bytedeco.openblas.global.openblas_nolapack.*;

/**
 * Thin wrapper around OpenBLAS (via JavaCPP/bytedeco) for CPU BLAS operations.
 * Falls back gracefully if OpenBLAS native libraries are not available.
 */
public class BlasOps {
    private static final boolean available;

    static {
        boolean ok = false;
        try {
            // Force class loading which triggers native lib extraction
            int order = CblasRowMajor;
            ok = true;
        } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
            System.err.println("BlasOps: OpenBLAS not available (" + e.getMessage() + "), using fallback.");
        }
        available = ok;
    }

    public static boolean isAvailable() {
        return available;
    }

    /**
     * C = alpha * A * B + beta * C  (row-major, no transpose)
     * A: [m x k], B: [k x n], C: [m x n]
     */
    public static void sgemm(float[] a, float[] b, float[] c, int m, int n, int k) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f, a, k,   // alpha=1, A, lda=k
                    b, n,          // B, ldb=n
                    0.0f, c, n);   // beta=0, C, ldc=n
    }
}
