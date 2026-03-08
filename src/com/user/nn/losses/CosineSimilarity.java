package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class CosineSimilarity extends Module {
    public int dim;
    public float eps;

    public CosineSimilarity(int dim, float eps) {
        this.dim = dim; this.eps = eps;
    }

    public CosineSimilarity() {
        this(1, 1e-8f);
    }

    public Tensor forward(Tensor x1, Tensor x2) {
        return Functional.cosine_similarity(x1, x2, dim, eps);
    }
}
