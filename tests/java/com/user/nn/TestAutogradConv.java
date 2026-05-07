package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.pooling.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradConv {

    @Test
    void testConv2dForwardBackward() {
        // 1 channel, 2 filters, 3x3 kernel, input 4x4, stride=1, pad=0
        int inC = 1, outC = 2, kh = 3, kw = 3, inH = 4, inW = 4;
        // Fix: stride should be 1, not inH/inW
        Conv2d conv = new Conv2d(inC, outC, kh, kw, 1, 1, 0, 0, true);

        // input: batch=1, flattened=1*4*4=16
        Tensor x = Torch.rand(new int[] { 1, inC * inH * inW });
        x.requires_grad = true;

        Tensor out = conv.forward(x);
        int outH = (inH + 2 * 0 - kh) / 1 + 1; // 2
        int outW = (inW + 2 * 0 - kw) / 1 + 1; // 2
        
        assertEquals(1, out.shape[0], "Conv2d batch size");
        int outSize = 1;
        for (int i = 1; i < out.shape.length; i++) outSize *= out.shape[i];
        assertEquals(outC * outH * outW, outSize, "Conv2d output size");

        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "Conv2d input grad exists");
        assertNotNull(conv.weight.getTensor().grad, "Conv2d weight grad exists");
        assertNotNull(conv.bias.getTensor().grad, "Conv2d bias grad exists");

        // Bias gradient should be outH*outW per output channel (sum of 1s over spatial*batch)
        Tensor bg = conv.bias.getTensor().grad;
        float expectedBiasGrad = (float) (outH * outW); // batch=1
        for (int oc = 0; oc < outC; oc++) {
            assertEquals(expectedBiasGrad, bg.data[oc], 1e-4f, 
                "Conv2d bias grad[" + oc + "] mismatch");
        }
    }

    @Test
    void testMaxPool2dBackward() {
        int inC = 1, inH = 4, inW = 4;
        MaxPool2d pool = new MaxPool2d(2, 2, 2, 2, 0, 0, inC, inH, inW);

        // Create a known input
        float[] data = new float[inC * inH * inW];
        for (int i = 0; i < data.length; i++) data[i] = i;
        Tensor x = Torch.tensor(data, 1, inC * inH * inW);
        x.requires_grad = true;

        Tensor out = pool.forward(x);
        int outH = 2, outW = 2;
        int outSize = 1;
        for (int i = 1; i < out.shape.length; i++) outSize *= out.shape[i];
        assertEquals(inC * outH * outW, outSize, "MaxPool output size");

        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "MaxPool input grad exists");
        // Gradient should only flow to max positions
        // For [0,1,2,3; 4,5,6,7; 8,9,10,11; 12,13,14,15]:
        // max positions: (1,1)=5, (1,3)=7, (3,1)=13, (3,3)=15
        assertEquals(1f, x.grad.data[5], "MaxPool grad at max pos (5)");
        assertEquals(1f, x.grad.data[7], "MaxPool grad at max pos (7)");
        assertEquals(1f, x.grad.data[13], "MaxPool grad at max pos (13)");
        assertEquals(1f, x.grad.data[15], "MaxPool grad at max pos (15)");
        
        // Non-max positions should be 0
        assertEquals(0f, x.grad.data[0], "MaxPool grad at non-max pos (0)");
        assertEquals(0f, x.grad.data[3], "MaxPool grad at non-max pos (3)");
    }

    @Test
    void testAvgPool2dBackward() {
        int inC = 1, inH = 4, inW = 4;
        AvgPool2d pool = new AvgPool2d(2, 2, 2, 2, 0, 0, inC, inH, inW);

        Tensor x = Torch.ones(1, inC * inH * inW);
        x.requires_grad = true;

        Tensor out = pool.forward(x);
        int outH = 2, outW = 2;
        int outSize = 1;
        for (int i = 1; i < out.shape.length; i++) outSize *= out.shape[i];
        assertEquals(inC * outH * outW, outSize, "AvgPool output size");
        
        // All inputs are 1, kernel=2x2, so avg=1.0
        for (int i = 0; i < out.data.length; i++) {
            assertEquals(1.0f, out.data[i], 1e-5f, "AvgPool out[" + i + "]");
        }

        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "AvgPool input grad exists");
        // Each input element contributes to exactly 1 output with weight 1/4
        for (int i = 0; i < x.grad.data.length; i++) {
            assertEquals(0.25f, x.grad.data[i], 1e-5f, "AvgPool grad[" + i + "]");
        }
    }

    @Test
    void testZeroPad2dBackward() {
        int inC = 1, inH = 2, inW = 2;
        ZeroPad2d pad = new ZeroPad2d(1, 1, inC, inH, inW);

        Tensor x = Torch.tensor(new float[] { 1, 2, 3, 4 }, 1, inC * inH * inW);
        x.requires_grad = true;

        Tensor out = pad.forward(x);
        int outH = 4, outW = 4;
        int outSize = 1;
        for (int i = 1; i < out.shape.length; i++) outSize *= out.shape[i];
        assertEquals(inC * outH * outW, outSize, "ZeroPad output size");
        
        assertEquals(0f, out.data[0], "ZeroPad corner is 0");
        assertEquals(1f, out.data[5], "ZeroPad inner (1,1) = 1");
        assertEquals(2f, out.data[6], "ZeroPad inner (1,2) = 2");
        assertEquals(3f, out.data[9], "ZeroPad inner (2,1) = 3");
        assertEquals(4f, out.data[10], "ZeroPad inner (2,2) = 4");

        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "ZeroPad input grad exists");
        for (int i = 0; i < 4; i++) {
            assertEquals(1f, x.grad.data[i], 1e-5f, "ZeroPad grad[" + i + "]");
        }
    }

    @Test
    void testConvTranspose2dBackward() {
        // inC=2, outC=1, kH=3, kW=3, inH=2, inW=2, stride=2, pad=0, outPad=1
        int inC = 2, outC = 1, kh = 3, kw = 3, inH2 = 2, inW2 = 2;
        ConvTranspose2d deconv = new ConvTranspose2d(inC, outC, kh, 2, 0, 1, true);

        Tensor x = Torch.ones(1, inC * inH2 * inW2);
        x.requires_grad = true;

        Tensor out = deconv.forward(x);
        int outH = 6, outW = 6;
        assertEquals(1, out.shape[0], "ConvTranspose2d batch size");
        int outSize = 1;
        for (int i = 1; i < out.shape.length; i++) outSize *= out.shape[i];
        assertEquals(outC * outH * outW, outSize, "ConvTranspose2d output size");

        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "ConvTranspose2d input grad exists");
        assertNotNull(deconv.weight.getTensor().grad, "ConvTranspose2d weight grad exists");
        assertNotNull(deconv.bias.getTensor().grad, "ConvTranspose2d bias grad exists");

        float expectedBiasGrad = (float) (outH * outW);
        assertEquals(expectedBiasGrad, deconv.bias.getTensor().grad.data[0], 1e-3f, "ConvTranspose2d bias grad mismatch");
    }

    @Test
    @Tag("gpu")
    void testConv2dForwardBackwardGPU() {
        if (!CUDAOps.isAvailable()) return;
        
        try (MemoryScope scope = new MemoryScope()) {
            int inC = 1, outC = 2, kh = 3, kw = 3, inH = 4, inW = 4;
            Conv2d conv = new Conv2d(inC, outC, kh, kw, 1, 1, 0, 0, true);
            conv.toGPU();

            Tensor x = Torch.rand(new int[] { 1, inC, inH, inW }).toGPU();
            x.requires_grad = true;

            Tensor out = conv.forward(x);
            assertTrue(out.isGPU());
            
            Tensor loss = Torch.sumTensor(out);
            loss.backward();

            assertNotNull(x.grad);
            assertTrue(x.grad.isGPU());
            assertNotNull(conv.weight.getTensor().grad);
            assertTrue(conv.weight.getTensor().grad.isGPU());
        }
    }
}
