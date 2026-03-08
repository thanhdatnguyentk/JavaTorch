package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.activations.*;

public class TestBatch1 {
    public static void main(String[] args) {
        testSoftmaxForward();
        testSoftmaxGrad();
        testLogSoftmaxForward();
        testLogSoftmaxGrad();
        testGeluForwardAndGrad();
        testEluForwardAndGrad();
        testSiluForwardAndGrad();
        testModuleWrappers();
        testFunctionalAPI();
        System.out.println("All Batch 1 tests PASSED!");
    }

    private static void check(String name, boolean cond) {
        if (cond) {
            System.out.println("  PASS: " + name);
        } else {
            System.err.println("  FAIL: " + name);
            System.exit(1);
        }
    }

    private static void testSoftmaxForward() {
        System.out.println("Testing Softmax forward...");
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 1, 3);
        Tensor out = Torch.softmax(x, 1);

        // sum should be 1
        float sum = Torch.sum(out);
        check("Softmax sum is 1", Math.abs(sum - 1.0f) < 1e-6f);

        // exp(1)/(exp(1)+exp(2)+exp(3))
        check("Softmax[0]", Math.abs(out.data[0] - 0.09003057f) < 1e-5f);
        check("Softmax[1]", Math.abs(out.data[1] - 0.24472848f) < 1e-5f);
        check("Softmax[2]", Math.abs(out.data[2] - 0.66524094f) < 1e-5f);

        // 1D softmax
        Tensor x1d = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        Tensor out1d = Torch.softmax(x1d, 0);
        check("Softmax 1D sum", Math.abs(Torch.sum(out1d) - 1.0f) < 1e-6f);
        check("Softmax 1D[0]", Math.abs(out1d.data[0] - 0.09003057f) < 1e-5f);
    }

    private static void testSoftmaxGrad() {
        System.out.println("Testing Softmax grad...");
        // To get a meaningful gradient, create a scalar loss: loss = softmax(x)[0]
        // We do this via element-wise mul with a mask and then sum
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.softmax(x, 0);
        // loss = out[0]*1 + out[1]*0 + out[2]*0 = out[0]
        Tensor mask = new Tensor(new float[]{1.0f, 0.0f, 0.0f}, 3);
        Tensor masked = Torch.mul(out, mask);
        Tensor loss = Torch.sumTensor(masked);
        loss.backward();

        // d(s0)/dx0 = s0*(1-s0) ≈ 0.09003*(1-0.09003) = 0.08192
        // d(s0)/dx1 = -s0*s1 ≈ -0.09003*0.24473 = -0.02204
        // d(s0)/dx2 = -s0*s2 ≈ -0.09003*0.66524 = -0.05988
        check("Softmax grad[0]", Math.abs(x.grad.data[0] - 0.08192f) < 1e-4f);
        check("Softmax grad[1]", Math.abs(x.grad.data[1] - (-0.02204f)) < 1e-4f);
        check("Softmax grad[2]", Math.abs(x.grad.data[2] - (-0.05988f)) < 1e-4f);
    }

    private static void testLogSoftmaxForward() {
        System.out.println("Testing LogSoftmax forward...");
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        Tensor out = Torch.log_softmax(x, 0);

        // log_softmax = x - log(sum(exp(x)))
        check("LogSoftmax[0]", Math.abs(out.data[0] - (float)Math.log(0.09003057)) < 1e-4f);
        // exp(log_softmax) should give softmax
        check("exp(LogSoftmax) sum", Math.abs(
            (float)(Math.exp(out.data[0]) + Math.exp(out.data[1]) + Math.exp(out.data[2])) - 1.0f) < 1e-5f);
    }

    private static void testLogSoftmaxGrad() {
        System.out.println("Testing LogSoftmax grad...");
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.log_softmax(x, 0);
        // loss = log_softmax[0]
        Tensor mask = new Tensor(new float[]{1.0f, 0.0f, 0.0f}, 3);
        Tensor masked = Torch.mul(out, mask);
        Tensor loss = Torch.sumTensor(masked);
        loss.backward();

        // d(log_s0)/dx0 = 1 - s0 = 1 - 0.09003 = 0.90997
        // d(log_s0)/dx1 = -s1 = -0.24473
        // d(log_s0)/dx2 = -s2 = -0.66524
        check("LogSoftmax grad[0]", Math.abs(x.grad.data[0] - 0.90997f) < 1e-4f);
        check("LogSoftmax grad[1]", Math.abs(x.grad.data[1] - (-0.24473f)) < 1e-4f);
        check("LogSoftmax grad[2]", Math.abs(x.grad.data[2] - (-0.66524f)) < 1e-4f);
    }

    private static void testGeluForwardAndGrad() {
        System.out.println("Testing GELU...");
        Tensor x = new Tensor(new float[]{0.0f, 1.0f, -1.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.gelu(x);

        // gelu(0) = 0
        check("GELU(0)=0", Math.abs(out.data[0]) < 1e-6f);
        // gelu(1) ≈ 0.8412
        check("GELU(1)≈0.8412", Math.abs(out.data[1] - 0.8412f) < 1e-3f);
        // gelu(-1) ≈ -0.1588
        check("GELU(-1)≈-0.1588", Math.abs(out.data[2] - (-0.1588f)) < 1e-3f);

        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        check("GELU grad exists", x.grad != null);
        // gelu'(0) = 0.5
        check("GELU grad(0)≈0.5", Math.abs(x.grad.data[0] - 0.5f) < 1e-3f);
    }

    private static void testEluForwardAndGrad() {
        System.out.println("Testing ELU...");
        Tensor x = new Tensor(new float[]{1.0f, -1.0f, 0.0f}, 3);
        x.requires_grad = true;
        Tensor out = Torch.elu(x, 1.0f);

        check("ELU(1)=1", out.data[0] == 1.0f);
        // elu(-1, alpha=1) = 1*(exp(-1)-1) = -0.6321
        check("ELU(-1)≈-0.6321", Math.abs(out.data[1] - (-0.63212055f)) < 1e-5f);
        check("ELU(0)=0", Math.abs(out.data[2]) < 1e-6f);

        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        // d(elu)/dx at x=1: 1
        check("ELU grad(1)=1", Math.abs(x.grad.data[0] - 1.0f) < 1e-6f);
        // d(elu)/dx at x=-1: alpha*exp(-1) = 0.3679
        check("ELU grad(-1)≈0.3679", Math.abs(x.grad.data[1] - 0.36787944f) < 1e-5f);
    }

    private static void testSiluForwardAndGrad() {
        System.out.println("Testing SiLU...");
        Tensor x = new Tensor(new float[]{0.0f, 1.0f}, 2);
        x.requires_grad = true;
        Tensor out = Torch.silu(x);

        // silu(0) = 0*sigmoid(0) = 0*0.5 = 0
        check("SiLU(0)=0", Math.abs(out.data[0]) < 1e-6f);
        // silu(1) = 1*sigmoid(1) = 0.73106
        check("SiLU(1)≈0.7311", Math.abs(out.data[1] - 0.73105858f) < 1e-5f);

        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        // silu'(0) = sig(0) + 0*sig(0)*(1-sig(0)) = 0.5
        check("SiLU grad(0)=0.5", Math.abs(x.grad.data[0] - 0.5f) < 1e-4f);
        // silu'(1) = sig(1) + 1*sig(1)*(1-sig(1)) = 0.73106 + 0.19661 = 0.92767
        check("SiLU grad(1)≈0.9277", Math.abs(x.grad.data[1] - 0.92767f) < 1e-4f);
    }

    private static void testModuleWrappers() {
        System.out.println("Testing NN Module wrappers...");
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);

        GELU gelu = new GELU();
        Tensor g = gelu.forward(x);
        check("NN.GELU forward", Math.abs(g.data[0] - 0.8412f) < 1e-3f); // gelu(1) ≈ 0.84

        ELU elu = new ELU();
        Tensor e = elu.forward(x);
        check("NN.ELU forward", e.data[0] == 1.0f);

        SiLU silu = new SiLU();
        Tensor s = silu.forward(x);
        check("NN.SiLU forward", s.data[0] > 0);

        Softmax softmax = new Softmax(0);
        Tensor sm = softmax.forward(x);
        check("NN.Softmax sum=1", Math.abs(Torch.sum(sm) - 1.0f) < 1e-6f);

        LogSoftmax logSoftmax = new LogSoftmax(0);
        Tensor lsm = logSoftmax.forward(x);
        check("NN.LogSoftmax values negative", lsm.data[0] < 0);
    }

    private static void testFunctionalAPI() {
        System.out.println("Testing NN.F functional API...");
        Tensor x = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);

        Tensor sm = Functional.softmax(x, 0);
        check("F.softmax sum=1", Math.abs(Torch.sum(sm) - 1.0f) < 1e-6f);

        Tensor lsm = Functional.log_softmax(x, 0);
        check("F.log_softmax values", lsm.data[0] < 0);

        Tensor g = Functional.gelu(x);
        check("F.gelu", g.data[0] > 0);

        Tensor e = Functional.elu(x, 1.0f);
        check("F.elu", e.data[0] == 1.0f);

        Tensor s = Functional.silu(x);
        check("F.silu", s.data[0] > 0);
    }
}
