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

extern "C"
__global__ void relu_backward(float *x, float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = x[i] > 0 ? dy[i] : 0;
    }
}

extern "C"
__global__ void leaky_relu_forward(float *data, float negative_slope, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        data[i] = v > 0 ? v : v * negative_slope;
    }
}

extern "C"
__global__ void leaky_relu_backward(float *x, float *dy, float *dx, float negative_slope, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = x[i] > 0 ? dy[i] : dy[i] * negative_slope;
    }
}

extern "C"
__global__ void sigmoid_backward(float *y, float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = y[i];
        dx[i] = dy[i] * val * (1.0f - val);
    }
}

extern "C"
__global__ void tanh_backward(float *y, float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = y[i];
        dx[i] = dy[i] * (1.0f - val * val);
    }

extern "C"
__global__ void bce_forward(float *input, float *target, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float h = input[i];
        float y = target[i];
        if (h < 1e-12f) h = 1e-12f;
        if (h > 1.0f - 1e-12f) h = 1.0f - 1e-12f;
        out[i] = -(y * logf(h) + (1.0f - y) * logf(1.0f - h));
    }
}

extern "C"
__global__ void bce_backward(float *input, float *target, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float h = input[i];
        float y = target[i];
        if (h < 1e-12f) h = 1e-12f;
        if (h > 1.0f - 1e-12f) h = 1.0f - 1e-12f;
        dx[i] = (h - y) / (h * (1.0f - h) + 1e-12f);
    }
}

extern "C"
__global__ void bce_logits_forward(float *input, float *target, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float y = target[i];
        if (x > 0) {
            out[i] = x * (1.0f - y) + logf(1.0f + expf(-x));
        } else {
            out[i] = -x * y + logf(1.0f + expf(x));
        }
    }
}

extern "C"
__global__ void bce_logits_backward(float *input, float *target, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = 1.0f / (1.0f + expf(-input[i]));
        dx[i] = sig - target[i];
    }

extern "C"
__global__ void exp_kernel(float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}

extern "C"
__global__ void log_kernel(float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        if (v < 1e-12f) v = 1e-12f;
        out[i] = logf(v);
    }
}
