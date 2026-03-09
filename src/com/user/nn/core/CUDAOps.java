package com.user.nn.core;

import jcuda.Pointer;
import jcuda.jcudnn.*;
import java.io.File;
import java.util.*;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnStatus.*;
import static jcuda.jcudnn.cudnnDataType.*;
import static jcuda.jcudnn.cudnnTensorFormat.*;
import static jcuda.jcudnn.cudnnActivationMode.*;
import static jcuda.jcudnn.cudnnConvolutionMode.*;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.*;
import static jcuda.jcudnn.cudnnPoolingMode.*;
import static jcuda.jcudnn.cudnnNanPropagation.*;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.jcublas.cublasComputeType.*;
import static jcuda.jcublas.cublasGemmAlgo.*;
import static jcuda.cudaDataType.*;
import static jcuda.jcudnn.cudnnMathType.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.*;
import jcuda.runtime.cudaStream_t;

public class CUDAOps {
    private static cublasHandle cublasHandle;
    private static cudnnHandle cudnnHandle;
    
    // Custom PTX Kernels
    private static CUmodule module;
    private static CUfunction addFunction;
    private static CUfunction subFunction;
    private static CUfunction mulFunction;
    private static CUfunction addScalarFunction;
    private static CUfunction mulScalarFunction;
    private static CUfunction reluBackwardFunction;
    private static CUfunction leakyReluForwardFunction;
    private static CUfunction leakyReluBackwardFunction;
    private static CUfunction sigmoidBackwardFunction;
    private static CUfunction tanhBackwardFunction;
    private static CUfunction bceForwardFunction;
    private static CUfunction bceBackwardFunction;
    private static CUfunction bceLogitsForwardFunction;
    private static CUfunction bceLogitsBackwardFunction;
    private static CUfunction expFunction;
    private static CUfunction logFunction;
    private static CUfunction embeddingForwardFunction;
    private static CUfunction embeddingBackwardFunction;
    private static CUfunction addInplaceFunction;
    private static CUfunction subInplaceFunction;
    private static CUfunction mulInplaceFunction;
    
    // CUDA Streams for pipelining
    private static cudaStream_t computeStream;
    private static cudaStream_t transferStream;
    
    private static boolean initialized = false;
    private static boolean cudaAvailable = false;

    /**
     * Quick check whether CUDA/cuDNN/cuBLAS initialization succeeded or is
     * available on this system. Returns true if initialization can be performed.
     */
    public static boolean isAvailable() {
        try {
            init();
        } catch (Throwable t) {
            // init catches exceptions internally but guard anyway
        }
        return cudaAvailable;
    }

    public static synchronized void init() {
        if (!initialized) {
            try {
                cublasHandle = new cublasHandle();
                cublasCreate(cublasHandle);

                cudnnHandle = new cudnnHandle();
                int status = cudnnCreate(cudnnHandle);
                if (status != CUDNN_STATUS_SUCCESS) {
                    System.err.println("Warning: cuDNN initialization failed: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
                }

                // Initialize Driver API for custom kernels
                JCudaDriver.setExceptionsEnabled(true);
                cuInit(0);

                try {
                    module = new CUmodule();
                    // Try multiple paths to find kernels.ptx
                    File ptxFile = null;
                    String[] candidates = {
                        "bin/kernels.ptx",
                        "kernels.ptx",
                        System.getProperty("user.dir") + "/bin/kernels.ptx",
                        System.getProperty("user.dir") + "\\bin\\kernels.ptx"
                    };
                    // Also try relative to the classpath
                    String cp = System.getProperty("java.class.path");
                    if (cp != null) {
                        for (String p : cp.split(File.pathSeparator)) {
                            File f = new File(p);
                            if (f.isDirectory()) {
                                File candidate = new File(f, "kernels.ptx");
                                if (candidate.exists()) { ptxFile = candidate; break; }
                            }
                        }
                    }
                    if (ptxFile == null) {
                        for (String path : candidates) {
                            File f = new File(path);
                            if (f.exists()) { ptxFile = f; break; }
                        }
                    }
                    if (ptxFile != null && ptxFile.exists()) {
                        cuModuleLoad(module, ptxFile.getAbsolutePath());
                        System.out.println("[CUDAOps] Loaded PTX kernels from: " + ptxFile.getAbsolutePath());
                    } else {
                        System.err.println("Warning: kernels.ptx not found. Searched: " + java.util.Arrays.toString(candidates));
                    }

                    // Safely try to get functions; some PTX builds may omit variants (eg. *_backward)
                    addFunction = safeGetFunction("add_tensors");
                    subFunction = safeGetFunction("sub_tensors");
                    mulFunction = safeGetFunction("mul_tensors");
                    addScalarFunction = safeGetFunction("add_scalar");
                    mulScalarFunction = safeGetFunction("mul_scalar");
                    reluBackwardFunction = safeGetFunction("relu_backward");
                    leakyReluForwardFunction = safeGetFunction("leaky_relu_forward");
                    leakyReluBackwardFunction = safeGetFunction("leaky_relu_backward");
                    sigmoidBackwardFunction = safeGetFunction("sigmoid_backward");
                    tanhBackwardFunction = safeGetFunction("tanh_backward");
                    bceForwardFunction = safeGetFunction("bce_forward");
                    bceBackwardFunction = safeGetFunction("bce_backward");
                    bceLogitsForwardFunction = safeGetFunction("bce_logits_forward");
                    bceLogitsBackwardFunction = safeGetFunction("bce_logits_backward");
                    expFunction = safeGetFunction("exp_kernel");
                    logFunction = safeGetFunction("log_kernel");
                    embeddingForwardFunction = safeGetFunction("embedding_forward");
                    embeddingBackwardFunction = safeGetFunction("embedding_backward");
                    addInplaceFunction = safeGetFunction("add_tensors_inplace");
                    subInplaceFunction = safeGetFunction("sub_tensors_inplace");
                    mulInplaceFunction = safeGetFunction("mul_tensors_inplace");
                } catch (Exception e) {
                    System.err.println("Warning: Could not load custom PTX kernels. GPU element-wise operations may fail.");
                    e.printStackTrace();
                }

                // Create CUDA Streams
                computeStream = new cudaStream_t();
                jcuda.runtime.JCuda.cudaStreamCreate(computeStream);

                transferStream = new cudaStream_t();
                jcuda.runtime.JCuda.cudaStreamCreate(transferStream);

                // Bind streams to cuBLAS and cuDNN handles
                jcuda.jcublas.JCublas2.cublasSetStream(cublasHandle, computeStream);
                cudnnSetStream(cudnnHandle, computeStream);

                // If we reached here without exception, mark CUDA available
                cudaAvailable = true;
            } catch (Throwable t) {
                System.err.println("Warning: CUDA initialization failed or not available on this system: " + t.getMessage());
                t.printStackTrace();
                cudaAvailable = false;
            } finally {
                initialized = true;
            }
        }
    }
    
    /**
     * Get the compute stream for synchronization.
     */
    public static cudaStream_t getComputeStream() {
        init();
        return computeStream;
    }
    
    /**
     * Get the transfer stream for async H2D memory copies.
     */
    public static cudaStream_t getTransferStream() {
        init();
        return transferStream;
    }

    /**
     * Asynchronous Host-to-Device memory copy on the transfer stream.
     * Uses pinned memory for best performance.
     */
    public static void memcpyHostToDeviceAsync(Pointer devicePtr, Pointer hostPtr, long bytes) {
        init();
        jcuda.runtime.JCuda.cudaMemcpyAsync(devicePtr, hostPtr, bytes, 
            jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, transferStream);
    }
    
    /**
     * Synchronize the compute stream (wait for all pending compute ops).
     */
    public static void syncComputeStream() {
        init();
        jcuda.runtime.JCuda.cudaStreamSynchronize(computeStream);
    }
    
    /**
     * Synchronize the transfer stream (wait for all pending transfers).
     */
    public static void syncTransferStream() {
        init();
        jcuda.runtime.JCuda.cudaStreamSynchronize(transferStream);
    }

    private static void launchElementwiseKernel(CUfunction function, Tensor a, Tensor b, Tensor out) {
        int n = a.numel();
        Pointer pa = Pointer.to(a.getDevicePointer());
        Pointer pb = Pointer.to(b.getDevicePointer());
        Pointer pout = Pointer.to(out.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        
        Pointer kernelParameters = Pointer.to(pa, pb, pout, pn);
        
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        
        cuLaunchKernel(function,
            gridSizeX,  1, 1,
            blockSizeX, 1, 1,
            0, null,
            kernelParameters, null);
            
        out.markDirtyOnGPU();
    }

    private static CUfunction safeGetFunction(String name) {
        try {
            if (module == null) return null;
            CUfunction fn = new CUfunction();
            cuModuleGetFunction(fn, module, name);
            return fn;
        } catch (Throwable t) {
            System.err.println("Warning: PTX function '" + name + "' not found in module: " + t.getMessage());
            return null;
        }
    }
    
    private static void launchScalarKernel(CUfunction function, Tensor x, float scalar, Tensor out) {
        int n = x.numel();
        Pointer px = Pointer.to(x.getDevicePointer());
        Pointer ps = Pointer.to(new float[]{scalar});
        Pointer pn = Pointer.to(new int[]{n});
        
        Pointer kernelParameters = Pointer.to(px, ps, pn);
        
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        
        cuLaunchKernel(function,
            gridSizeX,  1, 1,
            blockSizeX, 1, 1,
            0, null,
            kernelParameters, null);
            
        out.markDirtyOnGPU();
    }

    public static void add(Tensor a, Tensor b, Tensor out) {
        init();
        if (!cudaAvailable || addFunction == null) {
            // CPU fallback
            a.toCPU(); b.toCPU(); out.toCPU();
            int n = a.numel();
            for (int i = 0; i < n; i++) out.data[i] = a.data[i] + b.data[i];
            out.markDirtyOnCPU();
            return;
        }
        launchElementwiseKernel(addFunction, a, b, out);
    }
    
    public static void sub(Tensor a, Tensor b, Tensor out) {
        init();
        if (!cudaAvailable || subFunction == null) {
            // CPU fallback
            a.toCPU(); b.toCPU(); out.toCPU();
            int n = a.numel();
            for (int i = 0; i < n; i++) out.data[i] = a.data[i] - b.data[i];
            out.markDirtyOnCPU();
            return;
        }
        launchElementwiseKernel(subFunction, a, b, out);
    }
    
    public static void mul(Tensor a, Tensor b, Tensor out) {
        init();
        if (!cudaAvailable || mulFunction == null) {
            // CPU fallback
            a.toCPU(); b.toCPU(); out.toCPU();
            int n = a.numel();
            for (int i = 0; i < n; i++) out.data[i] = a.data[i] * b.data[i];
            out.markDirtyOnCPU();
            return;
        }
        launchElementwiseKernel(mulFunction, a, b, out);
    }
    
    public static void add(Tensor x, float scalar, Tensor out) {
        init();
        if (!cudaAvailable || addScalarFunction == null) {
            x.toCPU(); out.toCPU();
            int n = x.numel();
            for (int i = 0; i < n; i++) out.data[i] = x.data[i] + scalar;
            out.markDirtyOnCPU();
            return;
        }
        // The scalar kernel updates in-place, we should copy if x != out
        if (x != out) {
            cudaMemcpy(out.getDevicePointer(), x.getDevicePointer(), (long) x.numel() * jcuda.Sizeof.FLOAT, cudaMemcpyDeviceToDevice);
        }
        launchScalarKernel(addScalarFunction, out, scalar, out);
    }
    
    public static void mul(Tensor x, float scalar, Tensor out) {
        init();
        if (!cudaAvailable || mulScalarFunction == null) {
            x.toCPU(); out.toCPU();
            int n = x.numel();
            for (int i = 0; i < n; i++) out.data[i] = x.data[i] * scalar;
            out.markDirtyOnCPU();
            return;
        }
        if (x != out) {
            cudaMemcpy(out.getDevicePointer(), x.getDevicePointer(), (long) x.numel() * jcuda.Sizeof.FLOAT, cudaMemcpyDeviceToDevice);
        }
        launchScalarKernel(mulScalarFunction, out, scalar, out);
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
        if (!cudaAvailable) {
            // CPU fallback: perform matmul on host memory
            a.toCPU(); b.toCPU(); out.toCPU();
            int m = a.shape[0];
            int k = a.shape[1];
            int n = b.shape[1];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0f;
                    for (int t = 0; t < k; t++) {
                        sum += a.data[i * k + t] * b.data[t * n + j];
                    }
                    out.data[i * n + j] = sum;
                }
            }
            out.markDirtyOnCPU();
            return;
        }
        
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
        if (MixedPrecision.isEnabled()) {
            cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        n, m, k, 
                        pAlpha, pB, CUDA_R_32F, n, 
                        pA, CUDA_R_32F, k, 
                        pBeta, pC, CUDA_R_32F, n,
                        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        n, m, k, 
                        pAlpha, pB, n, 
                        pA, k, 
                        pBeta, pC, n);
        }
        
        out.markDirtyOnGPU();
    }

    /**
     * Batched matrix multiplication: out = a @ b
     * a: [B, m, k], b: [B, k, n] -> out: [B, m, n]
     */
    public static void bmm(Tensor a, Tensor b, Tensor out) {
        init();
        if (!cudaAvailable) {
            // CPU fallback for batched matmul
            a.toCPU(); b.toCPU(); out.toCPU();
            int bSize = a.shape[0];
            int m = a.shape[1];
            int k = a.shape[2];
            int n = b.shape[2];
            for (int bi = 0; bi < bSize; bi++) {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        float sum = 0f;
                        for (int t = 0; t < k; t++) {
                            sum += a.data[bi*m*k + i*k + t] * b.data[bi*k*n + t*n + j];
                        }
                        out.data[bi*m*n + i*n + j] = sum;
                    }
                }
            }
            out.markDirtyOnCPU();
            return;
        }
        if (a.dim() != 3 || b.dim() != 3) {
            throw new IllegalArgumentException("bmm supports 3D tensors [B, M, K] and [B, K, N]");
        }
        int bSize = a.shape[0];
        int m = a.shape[1];
        int k = a.shape[2];
        if (b.shape[1] != k) throw new IllegalArgumentException("bmm k dimension mismatch: " + k + " vs " + b.shape[1]);
        int n = b.shape[2];
        
        Pointer pA = a.getDevicePointer();
        Pointer pB = b.getDevicePointer();
        Pointer pC = out.getDevicePointer();

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        long strideA = (long)m * k;
        long strideB = (long)k * n;
        long strideC = (long)m * n;

        // Row-major trick: row-major a(m,k) * b(k,n) = c(m,n) is col-major b'(n,k) * a'(k,m) = c'(n,m)
        cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k,
                                  pAlpha, pB, n, strideB,
                                  pA, k, strideA,
                                  pBeta, pC, n, strideC,
                                  bSize);
        out.markDirtyOnGPU();
    }
    
    /**
     * Convolution 2D Forward using cuDNN
     */
    public static void conv2dForward(Tensor x, Tensor weight, Tensor bias, Tensor out,
                                     int inC, int inH, int inW,
                                     int kH, int kW,
                                     int outC, int outH, int outW,
                                     int padH, int padW, int strideH, int strideW) {
        init();
        int batch = x.shape[0];

        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor yDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor wDesc = new cudnnFilterDescriptor();
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();

        cudnnCreateTensorDescriptor(xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inC, inH, inW);

        cudnnCreateFilterDescriptor(wDesc);
        // cuDNN filter shape: [outC, inC, kH, kW]
        // Our weight shape is [inC*kH*kW, outC]. We might need to transpose or adjust.
        // Wait, our Conv2d weight is [ksz, outC] where ksz = inC*kH*kW.
        // cuDNN expects [outC, inC, kH, kW]. 
        // This is a major layout difference. 
        // We'll need to transpose the weights to GPU layout once and keep them there.
        
        cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kH, kW);

        cudnnCreateConvolutionDescriptor(convDesc);
        cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        
        if (MixedPrecision.isEnabled()) {
            cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        }

        cudnnCreateTensorDescriptor(yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, outC, outH, outW);

        // Find algorithm (or use heuristic)
        int algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; 
        
        long[] workspaceSizeInBytes = {0};
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xDesc, wDesc, convDesc, yDesc, algo, workspaceSizeInBytes);
        
        Pointer workspace = new Pointer();
        if (workspaceSizeInBytes[0] > 0) {
            cudaMalloc(workspace, workspaceSizeInBytes[0]);
        }

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        // We need to handle weight layout. 
        // For now, if weights are not in cuDNN layout, we might have slow-down or need to transpose.
        // Let's assume we handle transposition in Conv2d.forward if needed.
        
        cudnnConvolutionForward(cudnnHandle, pAlpha, xDesc, x.getDevicePointer(), wDesc, weight.getDevicePointer(), 
                                convDesc, algo, workspace, workspaceSizeInBytes[0], pBeta, yDesc, out.getDevicePointer());

        if (bias != null) {
            cudnnTensorDescriptor bDesc = new cudnnTensorDescriptor();
            cudnnCreateTensorDescriptor(bDesc);
            cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outC, 1, 1);
            cudnnAddTensor(cudnnHandle, pAlpha, bDesc, bias.getDevicePointer(), pAlpha, yDesc, out.getDevicePointer());
            cudnnDestroyTensorDescriptor(bDesc);
        }

        if (workspaceSizeInBytes[0] > 0) {
            cudaFree(workspace);
        }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);

        out.markDirtyOnGPU();
    }

    /**
     * Fused Conv2d + Bias + ReLU Forward using cudnnConvolutionBiasActivationForward.
     * Eliminates intermediate VRAM read/write between Conv, BiasAdd, and Activation.
     */
    public static void conv2dBiasReluForward(Tensor x, Tensor weight, Tensor bias, Tensor out,
                                             int inC, int inH, int inW,
                                             int kH, int kW,
                                             int outC, int outH, int outW,
                                             int padH, int padW, int strideH, int strideW) {
        init();
        int batch = x.shape[0];

        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor yDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor wDesc = new cudnnFilterDescriptor();
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
        jcuda.jcudnn.cudnnActivationDescriptor actDesc = new jcuda.jcudnn.cudnnActivationDescriptor();

        cudnnCreateTensorDescriptor(xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inC, inH, inW);

        cudnnCreateFilterDescriptor(wDesc);
        cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kH, kW);

        cudnnCreateConvolutionDescriptor(convDesc);
        cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        if (MixedPrecision.isEnabled()) {
            cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        }

        cudnnCreateTensorDescriptor(yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, outC, outH, outW);

        // Activation descriptor for ReLU
        cudnnCreateActivationDescriptor(actDesc);
        cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

        // Bias descriptor
        cudnnTensorDescriptor bDesc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(bDesc);
        cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outC, 1, 1);

        int algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        long[] workspaceSizeInBytes = {0};
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xDesc, wDesc, convDesc, yDesc, algo, workspaceSizeInBytes);

        Pointer workspace = new Pointer();
        if (workspaceSizeInBytes[0] > 0) {
            cudaMalloc(workspace, workspaceSizeInBytes[0]);
        }

        Pointer pAlpha1 = Pointer.to(new float[]{1.0f});
        Pointer pAlpha2 = Pointer.to(new float[]{0.0f});

        // z descriptor (for residual add, not used here, so we pass y itself with alpha2=0)
        cudnnTensorDescriptor zDesc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(zDesc);
        cudnnSetTensor4dDescriptor(zDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, outC, outH, outW);

        cudnnConvolutionBiasActivationForward(cudnnHandle,
            pAlpha1, xDesc, x.getDevicePointer(),
            wDesc, weight.getDevicePointer(),
            convDesc, algo, workspace, workspaceSizeInBytes[0],
            pAlpha2, zDesc, out.getDevicePointer(),       // z = out (alpha2=0 means z is ignored)
            bDesc, bias.getDevicePointer(),
            actDesc, yDesc, out.getDevicePointer());

        if (workspaceSizeInBytes[0] > 0) {
            cudaFree(workspace);
        }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(zDesc);
        cudnnDestroyTensorDescriptor(bDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyActivationDescriptor(actDesc);

        out.markDirtyOnGPU();
    }

    /**
     * Conv2d Backward Data using cuDNN.
     * Computes gradient w.r.t. input: dx = conv_backward_data(dy, w)
     * @param dy output gradient [batch, outC, outH, outW]
     * @param weight filter weights in cuDNN layout [outC, inC, kH, kW]
     * @param dx output: gradient w.r.t input [batch, inC, inH, inW]
     */
    public static void conv2dBackwardData(Tensor dy, Tensor weight, Tensor dx,
                                           int inC, int inH, int inW,
                                           int kH, int kW,
                                           int outC, int outH, int outW,
                                           int padH, int padW, int strideH, int strideW) {
        init();
        int batch = dy.shape[0];

        cudnnTensorDescriptor dxDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dyDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor wDesc = new cudnnFilterDescriptor();
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();

        cudnnCreateTensorDescriptor(dxDesc);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inC, inH, inW);

        cudnnCreateTensorDescriptor(dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, outC, outH, outW);

        cudnnCreateFilterDescriptor(wDesc);
        cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kH, kW);

        cudnnCreateConvolutionDescriptor(convDesc);
        cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        if (MixedPrecision.isEnabled()) {
            cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        }

        int algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

        long[] workspaceSizeInBytes = {0};
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, wDesc, dyDesc, convDesc, dxDesc, algo, workspaceSizeInBytes);

        Pointer workspace = new Pointer();
        if (workspaceSizeInBytes[0] > 0) {
            cudaMalloc(workspace, workspaceSizeInBytes[0]);
        }

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        cudnnConvolutionBackwardData(cudnnHandle,
            pAlpha, wDesc, weight.getDevicePointer(),
            dyDesc, dy.getDevicePointer(),
            convDesc, algo, workspace, workspaceSizeInBytes[0],
            pBeta, dxDesc, dx.getDevicePointer());

        if (workspaceSizeInBytes[0] > 0) {
            cudaFree(workspace);
        }

        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);

        dx.markDirtyOnGPU();
    }

    /**
     * Conv2d Backward Filter using cuDNN.
     * Computes gradient w.r.t. weights: dw = conv_backward_filter(x, dy)
     * @param x input tensor [batch, inC, inH, inW]
     * @param dy output gradient [batch, outC, outH, outW]
     * @param dw output: gradient w.r.t filter [outC, inC, kH, kW] in cuDNN layout
     */
    public static void conv2dBackwardFilter(Tensor x, Tensor dy, Tensor dw,
                                             int inC, int inH, int inW,
                                             int kH, int kW,
                                             int outC, int outH, int outW,
                                             int padH, int padW, int strideH, int strideW) {
        init();
        int batch = x.shape[0];

        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dyDesc = new cudnnTensorDescriptor();
        cudnnFilterDescriptor dwDesc = new cudnnFilterDescriptor();
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();

        cudnnCreateTensorDescriptor(xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inC, inH, inW);

        cudnnCreateTensorDescriptor(dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, outC, outH, outW);

        cudnnCreateFilterDescriptor(dwDesc);
        cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kH, kW);

        cudnnCreateConvolutionDescriptor(convDesc);
        cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        if (MixedPrecision.isEnabled()) {
            cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        }

        int algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        long[] workspaceSizeInBytes = {0};
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, xDesc, dyDesc, convDesc, dwDesc, algo, workspaceSizeInBytes);

        Pointer workspace = new Pointer();
        if (workspaceSizeInBytes[0] > 0) {
            cudaMalloc(workspace, workspaceSizeInBytes[0]);
        }

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        cudnnConvolutionBackwardFilter(cudnnHandle,
            pAlpha, xDesc, x.getDevicePointer(),
            dyDesc, dy.getDevicePointer(),
            convDesc, algo, workspace, workspaceSizeInBytes[0],
            pBeta, dwDesc, dw.getDevicePointer());

        if (workspaceSizeInBytes[0] > 0) {
            cudaFree(workspace);
        }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyFilterDescriptor(dwDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);

        dw.markDirtyOnGPU();
    }

    /**
     * Conv2d Backward Bias using cuDNN.
     * Computes gradient w.r.t. bias: db = sum over batch and spatial dims of dy
     * @param dy output gradient [batch, outC, outH, outW]
     * @param db output: gradient w.r.t bias [1, outC, 1, 1]
     */
    public static void conv2dBackwardBias(Tensor dy, Tensor db,
                                           int outC, int outH, int outW) {
        init();
        int batch = dy.shape[0];

        cudnnTensorDescriptor dyDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor dbDesc = new cudnnTensorDescriptor();

        cudnnCreateTensorDescriptor(dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, outC, outH, outW);

        cudnnCreateTensorDescriptor(dbDesc);
        cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outC, 1, 1);

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        cudnnConvolutionBackwardBias(cudnnHandle,
            pAlpha, dyDesc, dy.getDevicePointer(),
            pBeta, dbDesc, db.getDevicePointer());

        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyTensorDescriptor(dbDesc);

        db.markDirtyOnGPU();
    }

    /**
     * Max Pooling 2D Forward using cuDNN
     */
    public static void maxPool2dForward(Tensor x, Tensor out,
                                         int inC, int inH, int inW,
                                         int kH, int kW,
                                         int outH, int outW,
                                         int padH, int padW, int strideH, int strideW) {
        init();
        int batch = x.shape[0];

        cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor yDesc = new cudnnTensorDescriptor();
        jcuda.jcudnn.cudnnPoolingDescriptor poolDesc = new jcuda.jcudnn.cudnnPoolingDescriptor();

        cudnnCreateTensorDescriptor(xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inC, inH, inW);

        cudnnCreateTensorDescriptor(yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inC, outH, outW);

        cudnnCreatePoolingDescriptor(poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, kH, kW, padH, padW, strideH, strideW);

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        cudnnPoolingForward(cudnnHandle, poolDesc, pAlpha, xDesc, x.getDevicePointer(), pBeta, yDesc, out.getDevicePointer());

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyPoolingDescriptor(poolDesc);

        out.markDirtyOnGPU();
    }

    /**
     * ReLU Forward using cuDNN
     */
    public static void reluForward(Tensor x, Tensor out) {
        init();
        int n = x.numel();
        cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n, 1, 1);
        jcuda.jcudnn.cudnnActivationDescriptor actDesc = new jcuda.jcudnn.cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(actDesc);
        cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});
        cudnnActivationForward(cudnnHandle, actDesc, pAlpha, desc, x.getDevicePointer(), pBeta, desc, out.getDevicePointer());
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroyActivationDescriptor(actDesc);
        out.markDirtyOnGPU();
    }

    public static void sigmoidForward(Tensor x, Tensor out) {
        init();
        int n = x.numel();
        cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n, 1, 1);
        jcuda.jcudnn.cudnnActivationDescriptor actDesc = new jcuda.jcudnn.cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(actDesc);
        cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});
        cudnnActivationForward(cudnnHandle, actDesc, pAlpha, desc, x.getDevicePointer(), pBeta, desc, out.getDevicePointer());
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroyActivationDescriptor(actDesc);
        out.markDirtyOnGPU();
    }

    public static void tanhForward(Tensor x, Tensor out) {
        init();
        int n = x.numel();
        cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n, 1, 1);
        jcuda.jcudnn.cudnnActivationDescriptor actDesc = new jcuda.jcudnn.cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(actDesc);
        cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0);
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});
        cudnnActivationForward(cudnnHandle, actDesc, pAlpha, desc, x.getDevicePointer(), pBeta, desc, out.getDevicePointer());
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroyActivationDescriptor(actDesc);
        out.markDirtyOnGPU();
    }

    public static void reluBackward(Tensor x, Tensor dy, Tensor dx) {
        init();
        int n = x.numel();
        // Fallback to CPU implementation if kernel not available
        if (reluBackwardFunction == null) {
            x.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = (x.data[i] > 0f) ? dy.data[i] : 0f;
            }
            if (x.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
            return;
        }
        Pointer px = Pointer.to(x.getDevicePointer());
        Pointer pdy = Pointer.to(dy.getDevicePointer());
        Pointer pdx = Pointer.to(dx.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(px, pdy, pdx, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(reluBackwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            dx.markDirtyOnGPU();
        } catch (Throwable t) {
            // Fallback to CPU implementation if GPU kernel fails at runtime
            x.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = (x.data[i] > 0f) ? dy.data[i] : 0f;
            }
            if (x.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
        }
    }

    public static void leakyReluForward(Tensor x, Tensor out, float negativeSlope) {
        init();
        int n = x.numel();
        if (leakyReluForwardFunction == null) {
            x.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float v = x.data[i];
                out.data[i] = (v > 0f) ? v : v * negativeSlope;
            }
            if (x.isGPU()) out.toGPU(); else out.markDirtyOnCPU();
            return;
        }
        if (x != out) {
            cudaMemcpy(out.getDevicePointer(), x.getDevicePointer(), (long) x.numel() * jcuda.Sizeof.FLOAT, cudaMemcpyDeviceToDevice);
        }
        Pointer pout = Pointer.to(out.getDevicePointer());
        Pointer ps = Pointer.to(new float[]{negativeSlope});
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pout, ps, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(leakyReluForwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            out.markDirtyOnGPU();
        } catch (Throwable t) {
            x.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float v = x.data[i];
                out.data[i] = (v > 0f) ? v : v * negativeSlope;
            }
            if (x.isGPU()) out.toGPU(); else out.markDirtyOnCPU();
        }
    }

    public static void leakyReluBackward(Tensor x, Tensor dy, Tensor dx, float negativeSlope) {
        init();
        int n = x.numel();
        if (leakyReluBackwardFunction == null) {
            x.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = (x.data[i] > 0f) ? dy.data[i] : dy.data[i] * negativeSlope;
            }
            if (x.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
            return;
        }
        Pointer px = Pointer.to(x.getDevicePointer());
        Pointer pdy = Pointer.to(dy.getDevicePointer());
        Pointer pdx = Pointer.to(dx.getDevicePointer());
        Pointer ps = Pointer.to(new float[]{negativeSlope});
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(px, pdy, pdx, ps, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(leakyReluBackwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            dx.markDirtyOnGPU();
        } catch (Throwable t) {
            x.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = (x.data[i] > 0f) ? dy.data[i] : dy.data[i] * negativeSlope;
            }
            if (x.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
        }
    }

    public static void sigmoidBackward(Tensor y, Tensor dy, Tensor dx) {
        init();
        int n = y.numel();
        if (sigmoidBackwardFunction == null) {
            y.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = dy.data[i] * y.data[i] * (1f - y.data[i]);
            }
            if (y.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
            return;
        }
        Pointer py = Pointer.to(y.getDevicePointer());
        Pointer pdy = Pointer.to(dy.getDevicePointer());
        Pointer pdx = Pointer.to(dx.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(py, pdy, pdx, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(sigmoidBackwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            dx.markDirtyOnGPU();
        } catch (Throwable t) {
            y.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = dy.data[i] * y.data[i] * (1f - y.data[i]);
            }
            if (y.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
        }
    }

    public static void tanhBackward(Tensor y, Tensor dy, Tensor dx) {
        init();
        int n = y.numel();
        if (tanhBackwardFunction == null) {
            y.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = dy.data[i] * (1f - y.data[i] * y.data[i]);
            }
            if (y.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
            return;
        }
        Pointer py = Pointer.to(y.getDevicePointer());
        Pointer pdy = Pointer.to(dy.getDevicePointer());
        Pointer pdx = Pointer.to(dx.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(py, pdy, pdx, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(tanhBackwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            dx.markDirtyOnGPU();
        } catch (Throwable t) {
            y.toCPU(); dy.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                dx.data[i] = dy.data[i] * (1f - y.data[i] * y.data[i]);
            }
            if (y.isGPU() || dy.isGPU()) dx.toGPU(); else dx.markDirtyOnCPU();
        }
    }

    public static void bceForward(Tensor input, Tensor target, Tensor out) {
        init();
        int n = input.numel();
        if (bceForwardFunction == null) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float h = Math.max(1e-12f, Math.min(1f - 1e-12f, input.data[i]));
                float y = target.data[i];
                out.data[i] = -(y * (float) Math.log(h) + (1f - y) * (float) Math.log(1f - h));
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
            return;
        }
        Pointer pi = Pointer.to(input.getDevicePointer());
        Pointer pt = Pointer.to(target.getDevicePointer());
        Pointer pout = Pointer.to(out.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pi, pt, pout, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(bceForwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            out.markDirtyOnGPU();
        } catch (Throwable t) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float h = Math.max(1e-12f, Math.min(1f - 1e-12f, input.data[i]));
                float y = target.data[i];
                out.data[i] = -(y * (float) Math.log(h) + (1f - y) * (float) Math.log(1f - h));
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
        }
    }

    public static void bceBackward(Tensor input, Tensor target, Tensor dx) {
        init();
        int n = input.numel();
        if (bceBackwardFunction == null) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                float h = Math.max(1e-12f, Math.min(1f - 1e-12f, input.data[i]));
                float y = target.data[i];
                dx.data[i] = ((h - y) / (h * (1f - h) + 1e-12f));
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); dx.toGPU(); } else { dx.markDirtyOnCPU(); }
            return;
        }
        Pointer pi = Pointer.to(input.getDevicePointer());
        Pointer pt = Pointer.to(target.getDevicePointer());
        Pointer pdx = Pointer.to(dx.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pi, pt, pdx, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(bceBackwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            dx.markDirtyOnGPU();
        } catch (Throwable t) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                float h = Math.max(1e-12f, Math.min(1f - 1e-12f, input.data[i]));
                float y = target.data[i];
                dx.data[i] = ((h - y) / (h * (1f - h) + 1e-12f));
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); dx.toGPU(); } else { dx.markDirtyOnCPU(); }
        }
    }

    public static void bceLogitsForward(Tensor input, Tensor target, Tensor out) {
        init();
        int n = input.numel();
        if (bceLogitsForwardFunction == null) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float x = input.data[i];
                float y = target.data[i];
                if (x > 0) {
                    out.data[i] = x * (1 - y) + (float) Math.log(1 + Math.exp(-x));
                } else {
                    out.data[i] = -x * y + (float) Math.log(1 + Math.exp(x));
                }
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
            return;
        }
        Pointer pi = Pointer.to(input.getDevicePointer());
        Pointer pt = Pointer.to(target.getDevicePointer());
        Pointer pout = Pointer.to(out.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pi, pt, pout, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(bceLogitsForwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            out.markDirtyOnGPU();
        } catch (Throwable t) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float x = input.data[i];
                float y = target.data[i];
                if (x > 0) {
                    out.data[i] = x * (1 - y) + (float) Math.log(1 + Math.exp(-x));
                } else {
                    out.data[i] = -x * y + (float) Math.log(1 + Math.exp(x));
                }
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
        }
    }

    public static void bceLogitsBackward(Tensor input, Tensor target, Tensor dx) {
        init();
        int n = input.numel();
        if (bceLogitsBackwardFunction == null) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                float sig = 1.0f / (1.0f + (float) Math.exp(-input.data[i]));
                dx.data[i] = sig - target.data[i];
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); dx.toGPU(); } else { dx.markDirtyOnCPU(); }
            return;
        }
        Pointer pi = Pointer.to(input.getDevicePointer());
        Pointer pt = Pointer.to(target.getDevicePointer());
        Pointer pdx = Pointer.to(dx.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pi, pt, pdx, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(bceLogitsBackwardFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            dx.markDirtyOnGPU();
        } catch (Throwable t) {
            boolean wasGPU = input.isGPU() || target.isGPU();
            input.toCPU(); target.toCPU(); dx.toCPU();
            for (int i = 0; i < n; i++) {
                float sig = 1.0f / (1.0f + (float) Math.exp(-input.data[i]));
                dx.data[i] = sig - target.data[i];
            }
            if (wasGPU) { input.toGPU(); target.toGPU(); dx.toGPU(); } else { dx.markDirtyOnCPU(); }
        }
    }

    public static void exp(Tensor a, Tensor out) {
        init();
        int n = a.numel();
        if (expFunction == null) {
            boolean wasGPU = a.isGPU();
            a.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) out.data[i] = (float) Math.exp(a.data[i]);
            if (wasGPU) { a.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
            return;
        }
        Pointer pa = Pointer.to(a.getDevicePointer());
        Pointer pout = Pointer.to(out.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pa, pout, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(expFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            out.markDirtyOnGPU();
        } catch (Throwable t) {
            boolean wasGPU = a.isGPU();
            a.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) out.data[i] = (float) Math.exp(a.data[i]);
            if (wasGPU) { a.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
        }
    }

    public static void log(Tensor a, Tensor out) {
        init();
        int n = a.numel();
        if (logFunction == null) {
            boolean wasGPU = a.isGPU();
            a.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float v = Math.max(1e-12f, a.data[i]);
                out.data[i] = (float) Math.log(v);
            }
            if (wasGPU) { a.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
            return;
        }
        Pointer pa = Pointer.to(a.getDevicePointer());
        Pointer pout = Pointer.to(out.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParameters = Pointer.to(pa, pout, pn);
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
        try {
            cuLaunchKernel(logFunction, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, kernelParameters, null);
            out.markDirtyOnGPU();
        } catch (Throwable t) {
            boolean wasGPU = a.isGPU();
            a.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                float v = Math.max(1e-12f, a.data[i]);
                out.data[i] = (float) Math.log(v);
            }
            if (wasGPU) { a.toGPU(); out.toGPU(); } else { out.markDirtyOnCPU(); }
        }
    }

    /**
     * Transpose on GPU using cuBLAS sgeam
     * a: [m, n] -> out: [n, m]
     */
    public static void transpose(Tensor a, Tensor out) {
        init();
        int m = a.shape[0];
        int n = a.shape[1];
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});
        // sgeam: C = alpha * A' + beta * B'
        // To get C = A', we set beta = 0.
        jcuda.jcublas.JCublas2.cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                           n, m, pAlpha, a.getDevicePointer(), m, 
                                           pBeta, new Pointer(), n, 
                                           out.getDevicePointer(), n);
        out.markDirtyOnGPU();
    }

    /**
     * GPU-accelerated concat using cudaMemcpyDeviceToDevice.
     */
    public static void concat(java.util.List<Tensor> tensors, Tensor out, int dim) {
        init();
        int[] outShape = out.shape;
        int outerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= outShape[i];
        int totalDimSize = outShape[dim];
        int innerSize = 1;
        for (int i = dim + 1; i < outShape.length; i++) innerSize *= outShape[i];

        int currentOffset = 0;
        for (Tensor t : tensors) {
            int tDimSize = t.shape[dim];
            for (int i = 0; i < outerSize; i++) {
                long srcByteOff = (long) i * tDimSize * innerSize * 4;
                long dstByteOff = (long) (i * totalDimSize + currentOffset) * innerSize * 4;
                cudaMemcpy(out.getDevicePointer().withByteOffset(dstByteOff),
                           t.getDevicePointer().withByteOffset(srcByteOff),
                           (long) tDimSize * innerSize * 4, cudaMemcpyDeviceToDevice);
            }
            currentOffset += tDimSize;
        }
        out.markDirtyOnGPU();
    }

    /**
     * GPU-accelerated narrow using cudaMemcpyDeviceToDevice.
     */
    public static void narrow(Tensor input, Tensor out, int dim, int start, int length) {
        init();
        int outerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= input.shape[i];
        int innerSize = 1;
        for (int i = dim + 1; i < input.shape.length; i++) innerSize *= input.shape[i];
        int oldDimSize = input.shape[dim];

        for (int i = 0; i < outerSize; i++) {
            long srcByteOff = (long) (i * oldDimSize + start) * innerSize * 4;
            long dstByteOff = (long) i * length * innerSize * 4;
            cudaMemcpy(out.getDevicePointer().withByteOffset(dstByteOff),
                           input.getDevicePointer().withByteOffset(srcByteOff),
                           (long) length * innerSize * 4, cudaMemcpyDeviceToDevice);
        }
        out.markDirtyOnGPU();
    }

    public static void reduceSum(Tensor a, Tensor out) {
        init();
        int na = a.numel();
        int nout = out.numel();
        cudnnTensorDescriptor aDesc = new cudnnTensorDescriptor();
        cudnnTensorDescriptor outDesc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(aDesc);
        cudnnCreateTensorDescriptor(outDesc);
        cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, na, 1, 1);
        cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nout, 1, 1);

        cudnnReduceTensorDescriptor reduceDesc = new cudnnReduceTensorDescriptor();
        cudnnCreateReduceTensorDescriptor(reduceDesc);
        cudnnSetReduceTensorDescriptor(reduceDesc, 
            jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, 
            CUDNN_DATA_FLOAT, 
            CUDNN_PROPAGATE_NAN, 
            jcuda.jcudnn.cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, 
            jcuda.jcudnn.cudnnIndicesType.CUDNN_32BIT_INDICES);

        long[] workspaceSizeInBytes = {0};
        cudnnGetReductionWorkspaceSize(cudnnHandle, reduceDesc, aDesc, outDesc, workspaceSizeInBytes);
        Pointer workspace = new Pointer();
        if (workspaceSizeInBytes[0] > 0) {
            cudaMalloc(workspace, workspaceSizeInBytes[0]);
        }

        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        cudnnReduceTensor(cudnnHandle, reduceDesc, 
            null, 0, // Indices not used
            workspace, workspaceSizeInBytes[0], 
            pAlpha, aDesc, a.getDevicePointer(), 
            pBeta, outDesc, out.getDevicePointer());

        if (workspaceSizeInBytes[0] > 0) {
            cudaFree(workspace);
        }
        cudnnDestroyTensorDescriptor(aDesc);
        cudnnDestroyTensorDescriptor(outDesc);
        cudnnDestroyReduceTensorDescriptor(reduceDesc);
        out.markDirtyOnGPU();
    }

    // ---- Embedding GPU kernels ----

    /**
     * GPU embedding forward: out[i*d + col] = weight[indices[i]*d + col]
     * @param weight [num_embeddings, d]
     * @param indices [n] (float[] storing int indices)
     * @param out [n, d]
     */
    public static void embeddingForward(Tensor weight, Tensor indices, Tensor out) {
        init();
        int n = indices.numel();
        int d = weight.shape[1];
        if (!cudaAvailable || embeddingForwardFunction == null) {
            // CPU fallback
            weight.toCPU(); indices.toCPU(); out.toCPU();
            for (int i = 0; i < n; i++) {
                int idx = (int) indices.data[i];
                System.arraycopy(weight.data, idx * d, out.data, i * d, d);
            }
            out.markDirtyOnCPU();
            return;
        }
        int total = n * d;
        Pointer pWeight = Pointer.to(weight.getDevicePointer());
        Pointer pIndices = Pointer.to(indices.getDevicePointer());
        Pointer pOut = Pointer.to(out.getDevicePointer());
        Pointer pN = Pointer.to(new int[]{n});
        Pointer pD = Pointer.to(new int[]{d});
        Pointer kernelParams = Pointer.to(pWeight, pIndices, pOut, pN, pD);
        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) total / blockSize);
        cuLaunchKernel(embeddingForwardFunction,
            gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParams, null);
        out.markDirtyOnGPU();
    }

    /**
     * GPU embedding backward: atomicAdd grad_weight[indices[i]*d + col] += grad_out[i*d + col]
     * @param gradWeight [num_embeddings, d] — should be zeroed before call
     * @param indices [n]
     * @param gradOut [n, d]
     */
    public static void embeddingBackward(Tensor gradWeight, Tensor indices, Tensor gradOut) {
        init();
        int n = indices.numel();
        int d = gradWeight.shape[1];
        if (!cudaAvailable || embeddingBackwardFunction == null) {
            // CPU fallback
            gradWeight.toCPU(); indices.toCPU(); gradOut.toCPU();
            for (int i = 0; i < n; i++) {
                int idx = (int) indices.data[i];
                for (int j = 0; j < d; j++) {
                    gradWeight.data[idx * d + j] += gradOut.data[i * d + j];
                }
            }
            gradWeight.markDirtyOnCPU();
            return;
        }
        int total = n * d;
        Pointer pGW = Pointer.to(gradWeight.getDevicePointer());
        Pointer pIndices = Pointer.to(indices.getDevicePointer());
        Pointer pGO = Pointer.to(gradOut.getDevicePointer());
        Pointer pN = Pointer.to(new int[]{n});
        Pointer pD = Pointer.to(new int[]{d});
        Pointer kernelParams = Pointer.to(pGW, pIndices, pGO, pN, pD);
        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) total / blockSize);
        cuLaunchKernel(embeddingBackwardFunction,
            gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParams, null);
        gradWeight.markDirtyOnGPU();
    }

    // ---- In-place elementwise GPU ops ----

    private static void launchInplaceKernel(CUfunction function, Tensor a, Tensor b) {
        int n = a.numel();
        Pointer pa = Pointer.to(a.getDevicePointer());
        Pointer pb = Pointer.to(b.getDevicePointer());
        Pointer pn = Pointer.to(new int[]{n});
        Pointer kernelParams = Pointer.to(pa, pb, pn);
        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) n / blockSize);
        cuLaunchKernel(function,
            gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParams, null);
        a.markDirtyOnGPU();
    }

    public static void addInPlace(Tensor a, Tensor b) {
        init();
        if (!cudaAvailable || addInplaceFunction == null) {
            a.toCPU(); b.toCPU();
            for (int i = 0; i < a.numel(); i++) a.data[i] += b.data[i];
            a.markDirtyOnCPU();
            return;
        }
        launchInplaceKernel(addInplaceFunction, a, b);
    }

    public static void subInPlace(Tensor a, Tensor b) {
        init();
        if (!cudaAvailable || subInplaceFunction == null) {
            a.toCPU(); b.toCPU();
            for (int i = 0; i < a.numel(); i++) a.data[i] -= b.data[i];
            a.markDirtyOnCPU();
            return;
        }
        launchInplaceKernel(subInplaceFunction, a, b);
    }

    public static void mulInPlace(Tensor a, Tensor b) {
        init();
        if (!cudaAvailable || mulInplaceFunction == null) {
            a.toCPU(); b.toCPU();
            for (int i = 0; i < a.numel(); i++) a.data[i] *= b.data[i];
            a.markDirtyOnCPU();
            return;
        }
        launchInplaceKernel(mulInplaceFunction, a, b);
    }

    public static void shutdown() {
        if (initialized) {
            cublasDestroy(cublasHandle);
            cudnnDestroy(cudnnHandle);
            initialized = false;
        }
    }
}
