package com.user.nn;

import com.user.nn.core.*;

public class TestGPUEmbedding {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; }
        else { failed++; System.out.println("FAIL: " + name); }
    }

    static boolean allClose(float[] a, float[] b, float tol) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++)
            if (Math.abs(a[i] - b[i]) > tol) return false;
        return true;
    }

    public static void main(String[] args) {
        if (!CUDAOps.isAvailable()) {
            System.out.println("CUDA not available, skipping TestGPUEmbedding");
            return;
        }

        // --- Forward test ---
        int numEmb = 5, embDim = 4;
        float[] wData = new float[numEmb * embDim];
        for (int i = 0; i < wData.length; i++) wData[i] = i * 0.1f;
        Tensor weight = new Tensor(wData, numEmb, embDim);

        float[] idxData = {0f, 2f, 4f, 1f, 2f}; // 5 indices, including duplicate 2
        Tensor indices = new Tensor(idxData, 5);

        // CPU reference
        Tensor cpuOut = new Tensor(5, embDim);
        for (int i = 0; i < 5; i++) {
            int idx = (int) idxData[i];
            System.arraycopy(wData, idx * embDim, cpuOut.data, i * embDim, embDim);
        }

        // GPU path
        weight.toGPU();
        indices.toGPU();
        Tensor gpuOut = Torch.embedding(weight, indices);
        gpuOut.toCPU();

        check("embedding_forward", allClose(cpuOut.data, gpuOut.data, 1e-5f));

        // --- Backward test ---
        Tensor weight2 = new Tensor(wData.clone(), numEmb, embDim);
        weight2.requires_grad = true;
        weight2.toGPU();
        Tensor indices2 = new Tensor(idxData.clone(), 5);
        indices2.toGPU();

        Tensor out2 = Torch.embedding(weight2, indices2);
        // Create gradient of ones
        Tensor gradOut = Torch.ones(5, embDim);
        gradOut.toGPU();
        out2.backward(gradOut);

        // CPU backward reference
        float[] expectedGrad = new float[numEmb * embDim];
        for (int i = 0; i < 5; i++) {
            int idx = (int) idxData[i];
            for (int j = 0; j < embDim; j++) {
                expectedGrad[idx * embDim + j] += 1.0f;
            }
        }
        // idx=0: 1 time, idx=1: 1 time, idx=2: 2 times, idx=3: 0, idx=4: 1 time
        weight2.grad.toCPU();
        check("embedding_backward", allClose(expectedGrad, weight2.grad.data, 1e-5f));

        // --- Duplicate indices (atomicAdd correctness) ---
        float[] idxDup = {3f, 3f, 3f, 3f};
        Tensor weight3 = new Tensor(wData.clone(), numEmb, embDim);
        weight3.requires_grad = true;
        weight3.toGPU();
        Tensor idxTensor = new Tensor(idxDup, 4);
        idxTensor.toGPU();
        Tensor out3 = Torch.embedding(weight3, idxTensor);
        Tensor gradOut3 = Torch.ones(4, embDim);
        gradOut3.toGPU();
        out3.backward(gradOut3);
        weight3.grad.toCPU();
        // All 4 indices point to row 3, so row 3 grad should be 4.0 for each dim
        boolean dupOk = true;
        for (int j = 0; j < embDim; j++) {
            if (Math.abs(weight3.grad.data[3 * embDim + j] - 4.0f) > 1e-4f) dupOk = false;
            // Other rows should be 0
            if (Math.abs(weight3.grad.data[0 * embDim + j]) > 1e-4f) dupOk = false;
        }
        check("embedding_backward_duplicate_indices", dupOk);

        System.out.println("TestGPUEmbedding: " + passed + " passed, " + failed + " failed.");
        if (failed > 0) System.exit(1);
    }
}
