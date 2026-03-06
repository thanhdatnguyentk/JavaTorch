package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestBatch3 {
    public static void main(String[] args) {
        testMaxPool1d();
        testAvgPool1d();
        testAdaptiveAvgPool2d();
        testPad();
        System.out.println("All Batch 3 tests PASSED!");
    }

    private static void check(String name, boolean cond) {
        if (cond) {
            System.out.println("  PASS: " + name);
        } else {
            System.err.println("  FAIL: " + name);
            System.exit(1);
        }
    }

    private static void testMaxPool1d() {
        System.out.println("Testing MaxPool1d...");
        // [C=1, L=4]
        Tensor x = new Tensor(new float[]{1f, 3f, 2f, 4f}, 1, 4);
        x.requires_grad = true;
        // kernel=2, stride=2, pad=0 -> outL = (4-2)/2 + 1 = 2
        Tensor out = NN.F.max_pool1d(x, 2, 2, 0);
        check("MaxPool1d value[0]", out.data[0] == 3f);
        check("MaxPool1d value[1]", out.data[1] == 4f);

        Torch.sumTensor(out).backward();
        check("MaxPool1d grad[1]", x.grad.data[1] == 1f);
        check("MaxPool1d grad[3]", x.grad.data[3] == 1f);
        check("MaxPool1d grad[0]", x.grad.data[0] == 0f);
    }

    private static void testAvgPool1d() {
        System.out.println("Testing AvgPool1d...");
        Tensor x = new Tensor(new float[]{1f, 3f, 2f, 4f}, 1, 4);
        x.requires_grad = true;
        Tensor out = NN.F.avg_pool1d(x, 2, 2, 0);
        // (1+3)/2 = 2, (2+4)/2 = 3
        check("AvgPool1d value[0]", out.data[0] == 2f);
        check("AvgPool1d value[1]", out.data[1] == 3f);

        Torch.sumTensor(out).backward();
        // grad = 1/kernel = 0.5
        for (int i = 0; i < 4; i++) {
            check("AvgPool1d grad[" + i + "]", Math.abs(x.grad.data[i] - 0.5f) < 1e-6f);
        }
    }

    private static void testAdaptiveAvgPool2d() {
        System.out.println("Testing AdaptiveAvgPool2d...");
        // [C=1, H=4, W=4]
        Tensor x = Torch.ones(1, 4, 4);
        x.requires_grad = true;
        // out [2, 2]
        Tensor out = NN.F.adaptive_avg_pool2d(x, 2, 2);
        check("Adaptive out shape", out.shape[out.shape.length-1] == 2);
        check("Adaptive value", out.data[0] == 1f);

        Torch.sumTensor(out).backward();
        // Each output cell is average of 4 input cells (2x2 area)
        // grad = 1.0 / (2*2) = 0.25
        for (int i = 0; i < 16; i++) {
            check("Adaptive grad[" + i + "]", x.grad.data[i] == 0.25f);
        }
    }

    private static void testPad() {
        System.out.println("Testing Pad...");
        Tensor x = Torch.ones(2, 2);
        x.requires_grad = true;
        // pad [left, right, top, bottom] = [1, 1, 1, 1]
        Tensor out = NN.F.pad(x, new int[]{1, 1, 1, 1}, "constant", 0f);
        // 2x2 -> 4x4
        check("Pad shape H", out.shape[0] == 4);
        check("Pad shape W", out.shape[1] == 4);
        check("Pad value center", out.data[out.offset(1, 1)] == 1f);
        check("Pad value edge", out.data[0] == 0f);

        Torch.sumTensor(out).backward();
        // grad at center should be 1.0
        for (int i = 0; i < 4; i++) {
            check("Pad grad[" + i + "]", x.grad.data[i] == 1f);
        }
    }
}
