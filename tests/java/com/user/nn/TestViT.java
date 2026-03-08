package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.models.cv.ViT;
import java.util.Arrays;

public class TestViT {
    public static void main(String[] args) {
        
        int imgSize = 32;
        int patchSize = 4;
        int embedDim = 64; // Smaller for test
        int depth = 4;
        int numHeads = 4;
        int mlpDim = 128;
        
        System.out.println("Initializing ViT...");
        ViT model = new ViT(imgSize, patchSize, 3, 10, embedDim, depth, numHeads, mlpDim, 0.1f);
        model.to(Tensor.Device.GPU);
        model.train();

        System.out.println("Generating dummy input...");
        Tensor x = Torch.randn(new int[]{2, 3, 32, 32}).to(Tensor.Device.GPU);
        
        System.out.println("Forwarding ViT...");
        long start = System.currentTimeMillis();
        Tensor out = model.forward(x);
        long end = System.currentTimeMillis();
        
        System.out.println("Output shape: " + Arrays.toString(out.shape));
        System.out.println("Forward time: " + (end - start) + "ms");
        
        System.out.println("Output values: " + out);
        
        System.out.println("Backwarding ViT...");
        start = System.currentTimeMillis();
        out.backward();
        end = System.currentTimeMillis();
        System.out.println("Backward time: " + (end - start) + "ms");
        
        System.out.println("Test PASSED! Vision Transformer is functional on GPU.");
    }
}
