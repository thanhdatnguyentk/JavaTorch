package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestAutogradConv {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradConv...");
        boolean allPassed = true;
        allPassed &= testConv2dForwardBackward();
        allPassed &= testMaxPool2dBackward();
        allPassed &= testAvgPool2dBackward();
        allPassed &= testZeroPad2dBackward();
        allPassed &= testConvTranspose2dBackward();

        if (allPassed) {
            System.out.println("TestAutogradConv PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestAutogradConv FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testConv2dForwardBackward() {
        try {
            NN outer = new NN();
            // 1 channel, 2 filters, 3x3 kernel, input 4x4, stride=1, pad=0
            int inC = 1, outC = 2, kh = 3, kw = 3, inH = 4, inW = 4;
            NN.Conv2d conv = new NN.Conv2d(outer, inC, outC, kh, kw, inH, inW, 1, 0, true);

            // input: batch=1, flattened=1*4*4=16
            Tensor x = Torch.rand(new int[] { 1, inC * inH * inW });
            x.requires_grad = true;

            Tensor out = conv.forward(x);
            int outH = (inH - kh) / 1 + 1; // 2
            int outW = (inW - kw) / 1 + 1; // 2
            check(out.shape[0] == 1, "Conv2d batch size");
            check(out.shape[1] == outC * outH * outW, "Conv2d output size");

            Tensor loss = Torch.sumTensor(out);
            loss.backward();

            check(x.grad != null, "Conv2d input grad exists");
            check(conv.weight.getTensor().grad != null, "Conv2d weight grad exists");
            check(conv.bias.getTensor().grad != null, "Conv2d bias grad exists");

            // Bias gradient should be outH*outW per output channel (sum of 1s over
            // spatial*batch)
            Tensor bg = conv.bias.getTensor().grad;
            float expectedBiasGrad = (float) (outH * outW); // batch=1
            for (int oc = 0; oc < outC; oc++) {
                check(Math.abs(bg.data[oc] - expectedBiasGrad) < 1e-4f,
                        "Conv2d bias grad[" + oc + "] expected " + expectedBiasGrad + " got " + bg.data[oc]);
            }

            System.out.println("  Conv2d forward+backward OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testMaxPool2dBackward() {
        try {
            int inC = 1, inH = 4, inW = 4;
            NN.MaxPool2d pool = new NN.MaxPool2d(2, 2, 2, 2, 0, 0, inC, inH, inW);

            // Create a known input
            float[] data = new float[inC * inH * inW];
            for (int i = 0; i < data.length; i++)
                data[i] = i;
            Tensor x = Torch.tensor(data, 1, inC * inH * inW);
            x.requires_grad = true;

            Tensor out = pool.forward(x);
            int outH = 2, outW = 2;
            check(out.shape[1] == inC * outH * outW, "MaxPool output size");

            Tensor loss = Torch.sumTensor(out);
            loss.backward();

            check(x.grad != null, "MaxPool input grad exists");
            // Gradient should only flow to max positions
            // For [0,1,2,3; 4,5,6,7; 8,9,10,11; 12,13,14,15]:
            // max positions: (1,1)=5, (1,3)=7, (3,1)=13, (3,3)=15
            check(x.grad.data[5] == 1f, "MaxPool grad at max pos (5)");
            check(x.grad.data[7] == 1f, "MaxPool grad at max pos (7)");
            check(x.grad.data[13] == 1f, "MaxPool grad at max pos (13)");
            check(x.grad.data[15] == 1f, "MaxPool grad at max pos (15)");
            // Non-max positions should be 0
            check(x.grad.data[0] == 0f, "MaxPool grad at non-max pos (0)");
            check(x.grad.data[3] == 0f, "MaxPool grad at non-max pos (3)");

            System.out.println("  MaxPool2d backward OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testAvgPool2dBackward() {
        try {
            int inC = 1, inH = 4, inW = 4;
            NN.AvgPool2d pool = new NN.AvgPool2d(2, 2, 2, 2, 0, 0, inC, inH, inW);

            Tensor x = Torch.ones(1, inC * inH * inW);
            x.requires_grad = true;

            Tensor out = pool.forward(x);
            int outH = 2, outW = 2;
            check(out.shape[1] == inC * outH * outW, "AvgPool output size");
            // All inputs are 1, kernel=2x2, so avg=1.0
            for (int i = 0; i < out.data.length; i++) {
                check(Math.abs(out.data[i] - 1.0f) < 1e-5f, "AvgPool out[" + i + "]");
            }

            Tensor loss = Torch.sumTensor(out);
            loss.backward();

            check(x.grad != null, "AvgPool input grad exists");
            // Each input element contributes to exactly 1 output with weight 1/4
            for (int i = 0; i < x.grad.data.length; i++) {
                check(Math.abs(x.grad.data[i] - 0.25f) < 1e-5f,
                        "AvgPool grad[" + i + "] expected 0.25 got " + x.grad.data[i]);
            }

            System.out.println("  AvgPool2d backward OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testZeroPad2dBackward() {
        try {
            int inC = 1, inH = 2, inW = 2;
            NN.ZeroPad2d pad = new NN.ZeroPad2d(1, 1, inC, inH, inW);

            Tensor x = Torch.tensor(new float[] { 1, 2, 3, 4 }, 1, inC * inH * inW);
            x.requires_grad = true;

            Tensor out = pad.forward(x);
            int outH = 4, outW = 4;
            check(out.shape[1] == inC * outH * outW, "ZeroPad output size");
            // Check padded values are 0 and inner values are correct
            check(out.data[0] == 0f, "ZeroPad corner is 0");
            check(out.data[5] == 1f, "ZeroPad inner (1,1) = 1");
            check(out.data[6] == 2f, "ZeroPad inner (1,2) = 2");
            check(out.data[9] == 3f, "ZeroPad inner (2,1) = 3");
            check(out.data[10] == 4f, "ZeroPad inner (2,2) = 4");

            Tensor loss = Torch.sumTensor(out);
            loss.backward();

            check(x.grad != null, "ZeroPad input grad exists");
            // All input positions get gradient=1 from sum
            for (int i = 0; i < 4; i++) {
                check(Math.abs(x.grad.data[i] - 1f) < 1e-5f,
                        "ZeroPad grad[" + i + "] expected 1 got " + x.grad.data[i]);
            }

            System.out.println("  ZeroPad2d backward OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testConvTranspose2dBackward() {
        try {
            NN outer = new NN();
            // inC=2, outC=1, kH=3, kW=3, inH=2, inW=2, stride=2, pad=0, outPad=1
            int inC = 2, outC = 1, kh = 3, kw = 3, inH2 = 2, inW2 = 2;
            NN.ConvTranspose2d deconv = new NN.ConvTranspose2d(outer, inC, outC, kh, kw, inH2, inW2, 2, 0, 1, true);

            Tensor x = Torch.ones(1, inC * inH2 * inW2);
            x.requires_grad = true;

            Tensor out = deconv.forward(x);
            // outH = (2-1)*2 - 0 + 3 + 1 = 6, outW = 6
            int outH = 6, outW = 6;
            check(out.shape[0] == 1, "ConvTranspose2d batch size");
            check(out.shape[1] == outC * outH * outW, "ConvTranspose2d output size = " + out.shape[1]);

            Tensor loss = Torch.sumTensor(out);
            loss.backward();

            check(x.grad != null, "ConvTranspose2d input grad exists");
            check(deconv.weight.getTensor().grad != null, "ConvTranspose2d weight grad exists");
            check(deconv.bias.getTensor().grad != null, "ConvTranspose2d bias grad exists");

            // Bias grad: sum of 1s over outH*outW = 36
            float expectedBiasGrad = (float) (outH * outW);
            check(Math.abs(deconv.bias.getTensor().grad.data[0] - expectedBiasGrad) < 1e-3f,
                    "ConvTranspose2d bias grad expected " + expectedBiasGrad +
                            " got " + deconv.bias.getTensor().grad.data[0]);

            System.out.println("  ConvTranspose2d forward+backward OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
