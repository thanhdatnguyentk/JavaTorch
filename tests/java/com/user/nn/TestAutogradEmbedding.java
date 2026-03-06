package com.user.nn;

public class TestAutogradEmbedding {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradEmbedding...");
        
        nn lib = new nn();
        // 5 embeddings, dim 3
        int numE = 5;
        int dim = 3;
        nn.Embedding emb = new nn.Embedding(lib, numE, dim);
        
        // Input indices: [2, 1] (shape [2])
        Tensor indices = Torch.tensor(new float[]{2f, 1f}, 2);
        
        // Forward
        Tensor out = emb.forward(indices);
        
        // out should be [2, 3]
        if (out.shape[0] != 2 || out.shape[1] != 3) {
            System.err.println("Embedding output shape mismatch: " + out.toString());
            System.exit(1);
        }
        
        // Backward
        Tensor w = emb.weight.getTensor();
        w.requires_grad = true;
        out.backward();
        
        if (w.grad == null) {
            System.err.println("Embedding weight grad is null");
            System.exit(2);
        }
        
        // Indices were 2 and 1. 
        // Gradient of sum(out) w.r.t out is all 1s.
        // Gradient w.r.t w[2] and w[1] should be all 1s.
        // Gradient w.r.t w[0], w[3], w[4] should be all 0s.
        
        for (int i = 0; i < numE; i++) {
            for (int d = 0; d < dim; d++) {
                float g = w.grad.data[i * dim + d];
                float expected = (i == 1 || i == 2) ? 1.0f : 0.0f;
                if (Math.abs(g - expected) > 1e-6f) {
                    System.err.println("Gradient mismatch at index [" + i + "," + d + "]: expected " + expected + " got " + g);
                    System.exit(3);
                }
            }
        }
        
        System.out.println("TestAutogradEmbedding PASSED.");
    }
}
