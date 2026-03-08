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
    
    // CUDA Streams for pipelining
    private static cudaStream_t computeStream;
    private static cudaStream_t transferStream;
    
    private static boolean initialized = false;

    public static synchronized void init() {
        if (!initialized) {
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
                
                addFunction = new CUfunction();
                cuModuleGetFunction(addFunction, module, "add_tensors");
                
                subFunction = new CUfunction();
                cuModuleGetFunction(subFunction, module, "sub_tensors");
                
                mulFunction = new CUfunction();
                cuModuleGetFunction(mulFunction, module, "mul_tensors");
                
                addScalarFunction = new CUfunction();
                cuModuleGetFunction(addScalarFunction, module, "add_scalar");
                
                mulScalarFunction = new CUfunction();
                cuModuleGetFunction(mulScalarFunction, module, "mul_scalar");
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
            
            initialized = true;
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
        launchElementwiseKernel(addFunction, a, b, out);
    }
    
    public static void sub(Tensor a, Tensor b, Tensor out) {
        init();
        launchElementwiseKernel(subFunction, a, b, out);
    }
    
    public static void mul(Tensor a, Tensor b, Tensor out) {
        init();
        launchElementwiseKernel(mulFunction, a, b, out);
    }
    
    public static void add(Tensor x, float scalar, Tensor out) {
        init();
        // The scalar kernel updates in-place, we should copy if x != out
        if (x != out) {
            cudaMemcpy(out.getDevicePointer(), x.getDevicePointer(), (long) x.numel() * jcuda.Sizeof.FLOAT, cudaMemcpyDeviceToDevice);
        }
        launchScalarKernel(addScalarFunction, out, scalar, out);
    }
    
    public static void mul(Tensor x, float scalar, Tensor out) {
        init();
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

    public static void shutdown() {
        if (initialized) {
            cublasDestroy(cublasHandle);
            cudnnDestroy(cudnnHandle);
            initialized = false;
        }
    }
}
