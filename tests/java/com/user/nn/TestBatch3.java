package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestBatch3 {

    @Test
    void testMaxPool1d() {
        // [C=1, L=4]
        Tensor x = new Tensor(new float[]{1f, 3f, 2f, 4f}, 1, 4);
        x.requires_grad = true;
        // kernel=2, stride=2, pad=0 -> outL = (4-2)/2 + 1 = 2
        Tensor out = Functional.max_pool1d(x, 2, 2, 0);
        assertEquals(3f, out.data[0]);
        assertEquals(4f, out.data[1]);

        Torch.sumTensor(out).backward();
        assertEquals(1f, x.grad.data[1]);
        assertEquals(1f, x.grad.data[3]);
        assertEquals(0f, x.grad.data[0]);
    }

    @Test
    void testAvgPool1d() {
        Tensor x = new Tensor(new float[]{1f, 3f, 2f, 4f}, 1, 4);
        x.requires_grad = true;
        Tensor out = Functional.avg_pool1d(x, 2, 2, 0);
        // (1+3)/2 = 2, (2+4)/2 = 3
        assertEquals(2f, out.data[0]);
        assertEquals(3f, out.data[1]);

        Torch.sumTensor(out).backward();
        // grad = 1/kernel = 0.5
        for (int i = 0; i < 4; i++) {
            assertEquals(0.5f, x.grad.data[i], 1e-6f);
        }
    }

    @Test
    void testAdaptiveAvgPool2d() {
        // [C=1, H=4, W=4]
        Tensor x = Torch.ones(1, 4, 4);
        x.requires_grad = true;
        // out [2, 2]
        Tensor out = Functional.adaptive_avg_pool2d(x, 2, 2);
        assertEquals(2, out.shape[out.shape.length - 1]);
        assertEquals(1f, out.data[0]);

        Torch.sumTensor(out).backward();
        // Each output cell is average of 4 input cells (2x2 area)
        // grad = 1.0 / (2*2) = 0.25
        for (int i = 0; i < 16; i++) {
            assertEquals(0.25f, x.grad.data[i]);
        }
    }

    @Test
    void testPad() {
        Tensor x = Torch.ones(2, 2);
        x.requires_grad = true;
        // pad [left, right, top, bottom] = [1, 1, 1, 1]
        Tensor out = Functional.pad(x, new int[]{1, 1, 1, 1}, "constant", 0f);
        // 2x2 -> 4x4
        assertEquals(4, out.shape[0]);
        assertEquals(4, out.shape[1]);
        assertEquals(1f, out.data[out.offset(1, 1)]);
        assertEquals(0f, out.data[0]);

        Torch.sumTensor(out).backward();
        // grad at center should be 1.0
        for (int i = 0; i < 4; i++) {
            assertEquals(1f, x.grad.data[i]);
        }
    }
}
