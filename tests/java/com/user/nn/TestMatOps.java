package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestMatOps {

    @Test
    void testMatDot() {
        NN.Mat a = NN.mat_alloc(2, 3);
        NN.Mat b = NN.mat_alloc(3, 2);
        NN.mat_rand_seed(a, 1L, -1f, 1f);
        NN.mat_rand_seed(b, 2L, -1f, 1f);

        NN.Mat out = NN.mat_alloc(2, 2);
        NN.mat_dot(out, a, b);
        
        assertEquals(2, out.rows);
        assertEquals(2, out.cols);
    }

    @Test
    void testSumSub() {
        NN.Mat a = NN.mat_alloc(2, 2);
        NN.mat_fill(a, 1.0f);
        NN.Mat b = NN.mat_alloc(2, 2);
        NN.mat_fill(b, 2.0f);

        NN.Mat res = NN.mat_alloc(2, 2);
        System.arraycopy(a.es, 0, res.es, 0, a.es.length);
        
        NN.mat_sum(res, b); // res = 1 + 2 = 3
        for (float v : res.es) assertEquals(3.0f, v, 1e-6f);

        NN.mat_sub(res, b); // res = 3 - 2 = 1
        for (float v : res.es) assertEquals(1.0f, v, 1e-6f);
    }

    @Test
    void testApplyInplace() {
        NN.Mat m = NN.mat_alloc(2, 2);
        NN.mat_fill(m, 5.0f);
        NN.mat_apply_inplace(m, x -> x * 0f);
        for (float v : m.es) assertEquals(0.0f, v, 1e-6f);
    }
}
