package com.user.nn.core;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.runtime.JCuda.*;

public class CUDAOps {
    private static cublasHandle handle;
    private static boolean initialized = false;

    public static synchronized void init() {
        if (!initialized) {
            handle = new cublasHandle();
            cublasCreate(handle);
            initialized = true;
        }
    }

    /**
     * Matrix multiplication out = a * b (Row-major compatible using cuBLAS trick)
     * a: [m, k], b: [k, n] -> out: [m, n]
     */
    public static void matmul(Tensor a, Tensor b, Tensor out) {
        if (a.dim() != 2 || b.dim() != 2) {
            throw new IllegalArgumentException("Only 2D matmul supported on GPU currently");
        }
        init();
        
        int m = a.shape[0];
        int k = a.shape[1];
        int n = b.shape[1];

        Pointer pA = a.getDevicePointer();
        Pointer pB = b.getDevicePointer();
        Pointer pC = out.getDevicePointer();

        // alpha and beta must be on host or device based on cublasSetPointerMode
        // Default is host.
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        // Row-major A(m,k) * B(k,n) = C(m,n) is equivalent to
        // Column-major B'(n,k) * A'(k,m) = C'(n,m)
        // cublasSgemm(handle, transa, transb, m, n, k, ...) 
        // Using CM: Sgemm(handle, N, N, n, m, k, alpha, B, n, A, k, beta, C, n)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k, 
                    pAlpha, pB, n, 
                    pA, k, 
                    pBeta, pC, n);
        
        out.markDirtyOnGPU();
    }
    
    public static void shutdown() {
        if (initialized) {
            cublasDestroy(handle);
            initialized = false;
        }
    }
}
