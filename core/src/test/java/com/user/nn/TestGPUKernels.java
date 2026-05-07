package com.user.nn;

import com.user.nn.core.CUDAOps;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

public class TestGPUKernels {

    private void assertClose(Tensor t, float[] expected, float tol, String name) {
        t.toCPU();
        assertEquals(expected.length, t.data.length, name + " size mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], t.data[i], tol, name + " mismatch at index " + i);
        }
    }

    private float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    @Test
    @Tag("gpu")
    void testBasicOps() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");
        float tol = 1e-4f;

        Tensor a = Torch.tensor(new float[]{1f, 2f, -3f}, 3);
        Tensor b = Torch.tensor(new float[]{4f, -1f, 0.5f}, 3);
        Tensor out = Torch.zeros(3);
        a.toGPU(); b.toGPU(); out.toGPU();

        CUDAOps.add(a, b, out);
        assertClose(out, new float[]{5f, 1f, -2.5f}, tol, "add_tensors");
        
        CUDAOps.sub(a, b, out);
        assertClose(out, new float[]{-3f, 3f, -3.5f}, tol, "sub_tensors");
        
        CUDAOps.mul(a, b, out);
        assertClose(out, new float[]{4f, -2f, -1.5f}, tol, "mul_tensors");

        CUDAOps.add(a, 2f, out);
        assertClose(out, new float[]{3f, 4f, -1f}, tol, "add_scalar");
    }

    @Test
    @Tag("gpu")
    void testActivationKernels() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");
        float tol = 1e-4f;

        Tensor x = Torch.tensor(new float[]{-1f, 0f, 2f}, 3);
        Tensor dy = Torch.tensor(new float[]{0.1f, 0.2f, 0.3f}, 3);
        Tensor dx = Torch.zeros(3);
        x.toGPU(); dy.toGPU(); dx.toGPU();

        CUDAOps.reluBackward(x, dy, dx);
        assertClose(dx, new float[]{0f, 0f, 0.3f}, tol, "relu_backward");

        Tensor ySig = Torch.tensor(new float[]{0.2f, 0.8f}, 2);
        Tensor dySig = Torch.tensor(new float[]{1f, 1f}, 2);
        Tensor dxSig = Torch.zeros(2);
        ySig.toGPU(); dySig.toGPU(); dxSig.toGPU();
        CUDAOps.sigmoidBackward(ySig, dySig, dxSig);
        assertClose(dxSig, new float[]{0.16f, 0.16f}, tol, "sigmoid_backward");
    }

    @Test
    @Tag("gpu")
    void testLossKernels() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");
        float tol = 2e-4f;

        Tensor bceIn = Torch.tensor(new float[]{0.8f, 0.2f}, 2);
        Tensor bceTarget = Torch.tensor(new float[]{1f, 0f}, 2);
        Tensor bceOut = Torch.zeros(2);
        Tensor bceDx = Torch.zeros(2);
        bceIn.toGPU(); bceTarget.toGPU(); bceOut.toGPU(); bceDx.toGPU();

        CUDAOps.bceForward(bceIn, bceTarget, bceOut);
        float bceE = -(float) Math.log(0.8f);
        assertClose(bceOut, new float[]{bceE, bceE}, tol, "bce_forward");

        CUDAOps.bceBackward(bceIn, bceTarget, bceDx);
        float d0 = (0.8f - 1f) / (0.8f * 0.2f + 1e-12f);
        float d1 = (0.2f - 0f) / (0.2f * 0.8f + 1e-12f);
        assertClose(bceDx, new float[]{d0, d1}, tol, "bce_backward");
    }

    @Test
    @Tag("gpu")
    void testMathKernels() {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");
        float tol = 2e-4f;

        Tensor eIn = Torch.tensor(new float[]{-1f, 0f, 1f}, 3);
        Tensor eOut = Torch.zeros(3);
        eIn.toGPU(); eOut.toGPU();
        CUDAOps.exp(eIn, eOut);
        assertClose(eOut, new float[]{(float) Math.exp(-1), 1f, (float) Math.exp(1)}, tol, "exp_kernel");

        Tensor lIn = Torch.tensor(new float[]{1f, (float) Math.exp(1)}, 2);
        Tensor lOut = Torch.zeros(2);
        lIn.toGPU(); lOut.toGPU();
        CUDAOps.log(lIn, lOut);
        assertClose(lOut, new float[]{0f, 1f}, tol, "log_kernel");
    }
}
