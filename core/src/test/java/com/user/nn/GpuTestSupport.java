package com.user.nn;

import com.user.nn.core.CUDAOps;
import com.user.nn.core.BlasOps;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import org.junit.jupiter.api.Assumptions;

public final class GpuTestSupport {
    private GpuTestSupport() {
    }

    public static void assumeCuda() {
        Assumptions.assumeTrue(CUDAOps.isAvailable(), "CUDA unavailable - skipping GPU smoke test");
    }

    public static void seed(long seed) {
        Torch.manual_seed(seed);
    }

    public static void assumeBlas() {
        try {
            float[] a = {1f};
            float[] b = {1f};
            float[] c = {0f};
            if (!BlasOps.isAvailable()) {
                Assumptions.assumeTrue(false, "OpenBLAS unavailable - skipping BLAS-dependent GPU test");
            }
            BlasOps.sgemm(a, b, c, 1, 1, 1);
        } catch (Throwable t) {
            Assumptions.assumeTrue(false, "OpenBLAS probe failed - skipping BLAS-dependent GPU test: " + t.getMessage());
        }
    }

    public static void assertGpu(Tensor t, String name) {
        if (t == null) {
            throw new AssertionError(name + " tensor is null");
        }
        if (!t.isGPU()) {
            throw new AssertionError(name + " expected on GPU but was " + t.device);
        }
    }

    public static void assertFinite(Tensor t, String name) {
        if (t == null) {
            throw new AssertionError(name + " tensor is null");
        }
        t.toCPU();
        for (int i = 0; i < t.data.length; i++) {
            float v = t.data[i];
            if (!Float.isFinite(v)) {
                throw new AssertionError(name + " has non-finite value at index " + i + ": " + v);
            }
        }
    }
}
