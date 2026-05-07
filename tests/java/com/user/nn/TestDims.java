package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.norm.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class TestDims {
    @Test
    void testDimensionsAndShapes() {
        Tensor t1 = Torch.tensor(new float[3072], 3, 32, 32);
        assertArrayEquals(new int[]{3, 32, 32}, t1.shape);
        
        List<Tensor> list = new ArrayList<>();
        list.add(t1);
        list.add(t1);
        
        Tensor stacked = Torch.stack(list, 0);
        assertArrayEquals(new int[]{2, 3, 32, 32}, stacked.shape);
        
        Conv2d conv = new Conv2d(3, 64, 3, 3, 1, 1, 1, 1, false);
        Tensor out = conv.forward(stacked);
        assertArrayEquals(new int[]{2, 64, 32, 32}, out.shape);
        
        BatchNorm2d bn = new BatchNorm2d(64);
        Tensor bnOut = bn.forward(out);
        assertArrayEquals(new int[]{2, 64, 32, 32}, bnOut.shape);
    }
}
