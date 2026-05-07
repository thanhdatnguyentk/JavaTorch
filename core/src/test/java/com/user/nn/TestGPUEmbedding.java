package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

public class TestGPUEmbedding {

    private boolean allClose(float[] a, float[] b, float tol) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++)
            if (Math.abs(a[i] - b[i]) > tol) return false;
        return true;
    }

    @Test
    @Tag("gpu")
    void testForward() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        int numEmb = 5, embDim = 4;
        float[] wData = new float[numEmb * embDim];
        for (int i = 0; i < wData.length; i++) wData[i] = i * 0.1f;
        Tensor weight = new Tensor(wData, numEmb, embDim);

        float[] idxData = {0f, 2f, 4f, 1f, 2f};
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

        assertTrue(allClose(cpuOut.data, gpuOut.data, 1e-5f), "embedding_forward mismatch");
    }

    @Test
    @Tag("gpu")
    void testBackward() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        int numEmb = 5, embDim = 4;
        float[] wData = new float[numEmb * embDim];
        for (int i = 0; i < wData.length; i++) wData[i] = i * 0.1f;
        
        float[] idxData = {0f, 2f, 4f, 1f, 2f};
        Tensor weight2 = new Tensor(wData.clone(), numEmb, embDim);
        weight2.requires_grad = true;
        weight2.toGPU();
        Tensor indices2 = new Tensor(idxData.clone(), 5);
        indices2.toGPU();

        Tensor out2 = Torch.embedding(weight2, indices2);
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
        
        weight2.grad.toCPU();
        assertTrue(allClose(expectedGrad, weight2.grad.data, 1e-5f), "embedding_backward mismatch");
    }

    @Test
    @Tag("gpu")
    void testDuplicateIndicesAtomicAdd() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        int numEmb = 5, embDim = 4;
        float[] wData = new float[numEmb * embDim];
        float[] idxDup = {3f, 3f, 3f, 3f};
        
        Tensor weight = new Tensor(wData, numEmb, embDim);
        weight.requires_grad = true;
        weight.toGPU();
        Tensor idxTensor = new Tensor(idxDup, 4);
        idxTensor.toGPU();
        
        Tensor out = Torch.embedding(weight, idxTensor);
        Tensor gradOut = Torch.ones(4, embDim);
        gradOut.toGPU();
        out.backward(gradOut);
        
        weight.grad.toCPU();
        // Row 3 grad should be 4.0 for each dim
        for (int j = 0; j < embDim; j++) {
            assertEquals(4.0f, weight.grad.data[3 * embDim + j], 1e-4f);
            assertEquals(0.0f, weight.grad.data[0 * embDim + j], 1e-4f);
        }
    }
}
