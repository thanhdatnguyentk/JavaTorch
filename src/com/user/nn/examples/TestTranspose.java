package com.user.nn.examples;

import com.user.nn.core.*;

public class TestTranspose {
    public static void main(String[] args) {
        Tensor a = new Tensor(2, 3);
        a.data = new float[]{1, 2, 3, 4, 5, 6};
        a.toGPU();

        Tensor out = Torch.transpose(a, 0, 1);
        out.toCPU();

        System.out.println("Input: " + a);
        System.out.println("Output: " + out);
        
        // Expected: [1, 4, 2, 5, 3, 6]
        // Let's see what it actually prints!
    }
}
