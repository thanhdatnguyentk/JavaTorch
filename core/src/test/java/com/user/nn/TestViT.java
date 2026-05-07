package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.models.cv.ViT;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

public class TestViT {

    @Test
    @Tag("gpu")
    @Tag("slow")
    void testViTForwardBackwardGPU() {
        assumeTrue(Torch.hasGPU(), "GPU not available");

        int imgSize = 32;
        int patchSize = 4;
        int embedDim = 64;
        int depth = 2; // Reduced for test speed
        int numHeads = 4;
        int mlpDim = 128;
        
        ViT model = new ViT(imgSize, patchSize, 3, 10, embedDim, depth, numHeads, mlpDim, 0.1f);
        model.to(Tensor.Device.GPU);
        model.train();

        Tensor x = Torch.randn(new int[]{2, 3, 32, 32}).to(Tensor.Device.GPU);
        x.requires_grad = true;
        
        Tensor out = model.forward(x);
        assertArrayEquals(new int[]{2, 10}, out.shape);
        assertTrue(out.isGPU());

        out.backward();
        assertNotNull(x.grad, "ViT backward should populate input grad");
        
        // Memory cleanup
        model.toCPU(); // Free GPU mem for parameters
        x.close(); out.close();
    }

    @Test
    void testViTForwardCPU() {
        int imgSize = 16;
        int patchSize = 4;
        int embedDim = 32;
        int depth = 1;
        int numHeads = 2;
        int mlpDim = 64;
        
        ViT model = new ViT(imgSize, patchSize, 3, 10, embedDim, depth, numHeads, mlpDim, 0.0f);
        model.eval();

        Tensor x = Torch.randn(new int[]{1, 3, 16, 16});
        Tensor out = model.forward(x);
        
        assertArrayEquals(new int[]{1, 10}, out.shape);
        assertFalse(out.isGPU());
    }
}
