package com.user.nn.layers;

import com.user.nn.core.Module;
import com.user.nn.core.Tensor;

public class Reshape extends Module {
    private int[] targetShape;

    public Reshape(int... targetShape) {
        this.targetShape = targetShape;
    }

    @Override
    public Tensor forward(Tensor input) {
        // We might want to support -1 in targetShape eventually, 
        // but for now let's keep it simple.
        int batchSize = input.shape[0];
        int[] outShape = new int[targetShape.length + 1];
        outShape[0] = batchSize;
        System.arraycopy(targetShape, 0, outShape, 1, targetShape.length);
        
        return input.reshape(outShape);
    }
}
