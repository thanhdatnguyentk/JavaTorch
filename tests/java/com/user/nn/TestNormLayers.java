package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.norm.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestNormLayers {

    @Test
    void testLayerNorm() {
        int D = 4;
        LayerNorm ln = new LayerNorm(D);

        // batch=2, features=4
        Tensor x = Torch.tensor(new float[] {
                1f, 2f, 3f, 4f, // sample 0: mean=2.5, var=1.25
                4f, 4f, 4f, 4f // sample 1: all same => normalized=0
        }, 2, D);
        x.requires_grad = true;

        Tensor out = ln.forward(x);

        // Check sample 1: all values should be ~0 (since all inputs are the same)
        for (int d = 0; d < D; d++) {
            assertEquals(0.0f, out.data[1 * D + d], 1e-4f, "LayerNorm sample1 out[" + d + "] near 0");
        }

        // Check sample 0: mean of normalized should be ~0
        float mean = 0f;
        for (int d = 0; d < D; d++) mean += out.data[d];
        mean /= D;
        assertEquals(0.0f, mean, 1e-4f, "LayerNorm sample0 mean near 0");

        // backward
        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        assertNotNull(x.grad, "LayerNorm input grad exists");
        assertNotNull(ln.weight.getTensor().grad, "LayerNorm gamma grad exists");
        assertNotNull(ln.bias.getTensor().grad, "LayerNorm beta grad exists");

        // beta grad: each beta dimension gets sum over batch of 1 = 2
        for (int d = 0; d < D; d++) {
            assertEquals(2f, ln.bias.getTensor().grad.data[d], 1e-4f, "LayerNorm beta grad[" + d + "]");
        }
    }

    @Test
    void testInstanceNorm() {
        int C = 2, H = 2, W = 2;
        InstanceNorm in_ = new InstanceNorm(C, H, W);

        // batch=1, C=2, H=2, W=2 => flattened size = 8
        // channel 0: [1,2,3,4] => mean=2.5, normalized non-zero
        // channel 1: [5,5,5,5] => mean=5, normalized=0
        Tensor x = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 5, 5, 5 }, 1, C * H * W);
        x.requires_grad = true;

        Tensor out = in_.forward(x);

        // Check channel 1: all zeros
        for (int hw = 0; hw < H * W; hw++) {
            assertEquals(0.0f, out.data[C * H * W * 0 + 1 * H * W + hw], 1e-4f, "InstanceNorm channel1 out[" + hw + "] near 0");
        }

        // Check channel 0: mean ~0
        float mean = 0f;
        for (int hw = 0; hw < H * W; hw++) mean += out.data[hw];
        mean /= (H * W);
        assertEquals(0.0f, mean, 1e-4f, "InstanceNorm channel0 mean near 0");

        // backward
        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        assertNotNull(x.grad, "InstanceNorm input grad exists");
    }
}
