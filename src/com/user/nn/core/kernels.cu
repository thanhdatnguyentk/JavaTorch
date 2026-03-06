#include <cuda_fp16.h>

extern "C"
__global__ void add_scalar(float *data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += scalar;
    }
}

extern "C"
__global__ void mul_scalar(float *data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= scalar;
    }
}

extern "C"
__global__ void relu_forward(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

extern "C"
__global__ void sigmoid_forward(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

extern "C"
__global__ void tanh_forward(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = tanhf(data[i]);
    }
}

extern "C"
__global__ void add_tensors(float *a, float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C"
__global__ void sub_tensors(float *a, float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

extern "C"
__global__ void mul_tensors(float *a, float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

extern "C"
__global__ void add_tensors_fp16(half *a, half *b, half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hadd(a[i], b[i]);
    }
}

extern "C"
__global__ void sub_tensors_fp16(half *a, half *b, half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hsub(a[i], b[i]);
    }
}

extern "C"
__global__ void mul_tensors_fp16(half *a, half *b, half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hmul(a[i], b[i]);
    }
}

extern "C"
__global__ void fp32_to_fp16(float *in, half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

extern "C"
__global__ void fp16_to_fp32(half *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(in[i]);
    }
}

