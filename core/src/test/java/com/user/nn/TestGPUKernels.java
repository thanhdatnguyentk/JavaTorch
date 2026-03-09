package com.user.nn;

import com.user.nn.core.CUDAOps;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;

public class TestGPUKernels {
    private static void assertClose(Tensor t, float[] expected, float tol, String name) {
        t.toCPU();
        if (t.data.length != expected.length) {
            throw new RuntimeException(name + " size mismatch: got " + t.data.length + " expected " + expected.length);
        }
        for (int i = 0; i < expected.length; i++) {
            if (Math.abs(t.data[i] - expected[i]) > tol) {
                throw new RuntimeException(name + " mismatch at " + i + ": got " + t.data[i] + " expected " + expected[i]);
            }
        }
    }

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    public static void main(String[] args) {
        try {
            if (!CUDAOps.isAvailable()) {
                System.out.println("SKIP: CUDA/JCuda not available for TestGPUKernels");
                return;
            }

            float tol = 1e-4f;

            // Elementwise kernels
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

            // Scalar kernels
            CUDAOps.add(a, 2f, out);
            assertClose(out, new float[]{3f, 4f, -1f}, tol, "add_scalar");
            CUDAOps.mul(a, -2f, out);
            assertClose(out, new float[]{-2f, -4f, 6f}, tol, "mul_scalar");

            // Activation backward / forward kernels
            Tensor x = Torch.tensor(new float[]{-1f, 0f, 2f}, 3);
            Tensor dy = Torch.tensor(new float[]{0.1f, 0.2f, 0.3f}, 3);
            Tensor dx = Torch.zeros(3);
            x.toGPU(); dy.toGPU(); dx.toGPU();

            CUDAOps.reluBackward(x, dy, dx);
            assertClose(dx, new float[]{0f, 0f, 0.3f}, tol, "relu_backward");

            Tensor lrfIn = Torch.tensor(new float[]{-2f, 3f}, 2);
            Tensor lrfOut = Torch.zeros(2);
            lrfIn.toGPU(); lrfOut.toGPU();
            CUDAOps.leakyReluForward(lrfIn, lrfOut, 0.1f);
            assertClose(lrfOut, new float[]{-0.2f, 3f}, tol, "leaky_relu_forward");

            Tensor lrbDy = Torch.tensor(new float[]{0.5f, 0.5f}, 2);
            Tensor lrbDx = Torch.zeros(2);
            lrbDy.toGPU(); lrbDx.toGPU();
            CUDAOps.leakyReluBackward(lrfIn, lrbDy, lrbDx, 0.1f);
            assertClose(lrbDx, new float[]{0.05f, 0.5f}, tol, "leaky_relu_backward");

            Tensor ySig = Torch.tensor(new float[]{0.2f, 0.8f}, 2);
            Tensor dySig = Torch.tensor(new float[]{1f, 1f}, 2);
            Tensor dxSig = Torch.zeros(2);
            ySig.toGPU(); dySig.toGPU(); dxSig.toGPU();
            CUDAOps.sigmoidBackward(ySig, dySig, dxSig);
            assertClose(dxSig, new float[]{0.16f, 0.16f}, tol, "sigmoid_backward");

            Tensor yTanh = Torch.tensor(new float[]{0f, 0.5f}, 2);
            Tensor dyTanh = Torch.tensor(new float[]{1f, 2f}, 2);
            Tensor dxTanh = Torch.zeros(2);
            yTanh.toGPU(); dyTanh.toGPU(); dxTanh.toGPU();
            CUDAOps.tanhBackward(yTanh, dyTanh, dxTanh);
            assertClose(dxTanh, new float[]{1f, 1.5f}, tol, "tanh_backward");

            // BCE kernels
            Tensor bceIn = Torch.tensor(new float[]{0.8f, 0.2f}, 2);
            Tensor bceTarget = Torch.tensor(new float[]{1f, 0f}, 2);
            Tensor bceOut = Torch.zeros(2);
            Tensor bceDx = Torch.zeros(2);
            bceIn.toGPU(); bceTarget.toGPU(); bceOut.toGPU(); bceDx.toGPU();

            CUDAOps.bceForward(bceIn, bceTarget, bceOut);
            float bceE = -(float) Math.log(0.8f);
            assertClose(bceOut, new float[]{bceE, bceE}, 2e-4f, "bce_forward");

            CUDAOps.bceBackward(bceIn, bceTarget, bceDx);
            float d0 = (0.8f - 1f) / (0.8f * 0.2f + 1e-12f);
            float d1 = (0.2f - 0f) / (0.2f * 0.8f + 1e-12f);
            assertClose(bceDx, new float[]{d0, d1}, 2e-4f, "bce_backward");

            // BCE logits kernels
            Tensor logits = Torch.tensor(new float[]{2f, -2f}, 2);
            Tensor logitsTarget = Torch.tensor(new float[]{1f, 0f}, 2);
            Tensor logitsOut = Torch.zeros(2);
            Tensor logitsDx = Torch.zeros(2);
            logits.toGPU(); logitsTarget.toGPU(); logitsOut.toGPU(); logitsDx.toGPU();

            CUDAOps.bceLogitsForward(logits, logitsTarget, logitsOut);
            float e0 = (float) Math.log(1.0 + Math.exp(-2.0));
            float e1 = (float) Math.log(1.0 + Math.exp(-2.0));
            assertClose(logitsOut, new float[]{e0, e1}, 2e-4f, "bce_logits_forward");

            CUDAOps.bceLogitsBackward(logits, logitsTarget, logitsDx);
            float g0 = sigmoid(2f) - 1f;
            float g1 = sigmoid(-2f) - 0f;
            assertClose(logitsDx, new float[]{g0, g1}, 2e-4f, "bce_logits_backward");

            // exp/log kernels
            Tensor eIn = Torch.tensor(new float[]{-1f, 0f, 1f}, 3);
            Tensor eOut = Torch.zeros(3);
            eIn.toGPU(); eOut.toGPU();
            CUDAOps.exp(eIn, eOut);
            assertClose(eOut, new float[]{(float) Math.exp(-1), 1f, (float) Math.exp(1)}, 2e-4f, "exp_kernel");

            Tensor lIn = Torch.tensor(new float[]{1f, (float) Math.exp(1)}, 2);
            Tensor lOut = Torch.zeros(2);
            lIn.toGPU(); lOut.toGPU();
            CUDAOps.log(lIn, lOut);
            assertClose(lOut, new float[]{0f, 1f}, 2e-4f, "log_kernel");

            System.out.println("TestGPUKernels passed.");
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(1);
        }
    }
}
