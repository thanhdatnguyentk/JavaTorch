package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.Embedding;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestAutogradEmbedding {

    @Test
    void testEmbeddingForwardBackward() {
        // 5 embeddings, dim 3
        int numE = 5;
        int dim = 3;
        Embedding emb = new Embedding(numE, dim);
        
        // Input indices: [2, 1]
        Tensor indices = Torch.tensor(new float[]{2f, 1f}, 2);
        
        // Forward
        Tensor out = emb.forward(indices);
        assertArrayEquals(new int[]{2, 3}, out.shape, "Embedding output shape mismatch");
        
        // Backward
        Tensor w = emb.weight.getTensor();
        w.requires_grad = true;
        // Need to explicitly backward a gradient of ones to match the sum logic
        out.backward(); 
        
        assertNotNull(w.grad, "Embedding weight grad is null");
        
        // Indices were 2 and 1. 
        // Gradient of sum(out) w.r.t out is all 1s.
        // Gradient w.r.t w[2] and w[1] should be all 1s.
        // Gradient w.r.t w[0], w[3], w[4] should be all 0s.
        for (int i = 0; i < numE; i++) {
            for (int d = 0; d < dim; d++) {
                float g = w.grad.data[i * dim + d];
                float expected = (i == 1 || i == 2) ? 1.0f : 0.0f;
                assertEquals(expected, g, 1e-6f, "Gradient mismatch at index [" + i + "," + d + "]");
            }
        }
    }
}
