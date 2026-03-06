extern "C"
__global__ void add_one(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] + 1.0f;
    }
}
