package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.activations.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestBatch1 {

    @Test
    void testSoftmaxForward() {
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 1, 3);
        Tensor out = Torch.softmax(x, 1);

        assertEquals(1.0f, Torch.sum(out), 1e-6f, "Softmax sum should be 1");
        assertEquals(0.09003057f, out.data[0], 1e-5f, "Softmax[0] mismatch");
        assertEquals(0.24472848f, out.data[1], 1e-5f, "Softmax[1] mismatch");
        assertEquals(0.66524094f, out.data[2], 1e-5f, "Softmax[2] mismatch");

        // 1D softmax
        Tensor x1d = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        Tensor out1d = Torch.softmax(x1d, 0);
        assertEquals(1.0f, Torch.sum(out1d), 1e-6f, "Softmax 1D sum mismatch");
        assertEquals(0.09003057f, out1d.data[0], 1e-5f);
    }

    @Test
    void testSoftmaxGrad() {
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.softmax(x, 0);
        // loss = out[0]
        Tensor mask = new Tensor(new float[]{1.0f, 0.0f, 0.0f}, 3);
        Tensor masked = Torch.mul(out, mask);
        Tensor loss = Torch.sumTensor(masked);
        loss.backward();

        // d(s0)/dx0 = s0*(1-s0) ≈ 0.08192
        assertEquals(0.08192f, x.grad.data[0], 1e-4f);
        assertEquals(-0.02204f, x.grad.data[1], 1e-4f);
        assertEquals(-0.05988f, x.grad.data[2], 1e-4f);
    }

    @Test
    void testLogSoftmaxForward() {
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        Tensor out = Torch.log_softmax(x, 0);

        assertEquals((float)Math.log(0.09003057), out.data[0], 1e-4f);
        float sumExp = (float)(Math.exp(out.data[0]) + Math.exp(out.data[1]) + Math.exp(out.data[2]));
        assertEquals(1.0f, sumExp, 1e-5f, "exp(log_softmax) sum mismatch");
    }

    @Test
    void testLogSoftmaxGrad() {
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.log_softmax(x, 0);
        // loss = log_softmax[0]
        Tensor mask = new Tensor(new float[]{1.0f, 0.0f, 0.0f}, 3);
        Tensor masked = Torch.mul(out, mask);
        Tensor loss = Torch.sumTensor(masked);
        loss.backward();

        // d(log_s0)/dx0 = 1 - s0 = 1 - 0.09003 = 0.90997
        assertEquals(0.90997f, x.grad.data[0], 1e-4f);
        assertEquals(-0.24473f, x.grad.data[1], 1e-4f);
        assertEquals(-0.66524f, x.grad.data[2], 1e-4f);
    }

    @Test
    void testGeluForwardAndGrad() {
        Tensor x = new Tensor(new float[]{0.0f, 1.0f, -1.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.gelu(x);

        assertEquals(0.0f, out.data[0], 1e-6f);
        assertEquals(0.8412f, out.data[1], 1e-3f);
        assertEquals(-0.1588f, out.data[2], 1e-3f);

        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        assertNotNull(x.grad);
        assertEquals(0.5f, x.grad.data[0], 1e-3f);
    }

    @Test
    void testEluForwardAndGrad() {
        Tensor x = new Tensor(new float[]{1.0f, -1.0f, 0.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.elu(x, 1.0f);

        assertEquals(1.0f, out.data[0]);
        assertEquals(-0.63212055f, out.data[1], 1e-5f);
        assertEquals(0.0f, out.data[2], 1e-6f);

        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        assertEquals(1.0f, x.grad.data[0], 1e-6f);
        assertEquals(0.36787944f, x.grad.data[1], 1e-5f);
    }

    @Test
    void testSiluForwardAndGrad() {
        Tensor x = new Tensor(new float[]{0.0f, 1.0f}, 2);
        x.requires_grad = true;
        Tensor out = Torch.silu(x);

        assertEquals(0.0f, out.data[0], 1e-6f);
        assertEquals(0.73105858f, out.data[1], 1e-5f);

        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        assertEquals(0.5f, x.grad.data[0], 1e-4f);
        assertEquals(0.92767f, x.grad.data[1], 1e-4f);
    }

    @Test
    void testModuleWrappers() {
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);

        GELU gelu = new GELU();
        assertEquals(0.8412f, gelu.forward(x).data[0], 1e-3f);

        ELU elu = new ELU();
        assertEquals(1.0f, elu.forward(x).data[0]);

        SiLU silu = new SiLU();
        assertTrue(silu.forward(x).data[0] > 0);

        Softmax softmax = new Softmax(0);
        assertEquals(1.0f, Torch.sum(softmax.forward(x)), 1e-6f);

        LogSoftmax logSoftmax = new LogSoftmax(0);
        assertTrue(logSoftmax.forward(x).data[0] < 0);
    }

    @Test
    void testFunctionalAPI() {
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);

        assertEquals(1.0f, Torch.sum(Functional.softmax(x, 0)), 1e-6f);
        assertTrue(Functional.log_softmax(x, 0).data[0] < 0);
        assertTrue(Functional.gelu(x).data[0] > 0);
        assertEquals(1.0f, Functional.elu(x, 1.0f).data[0]);
        assertTrue(Functional.silu(x).data[0] > 0);
    }
}
