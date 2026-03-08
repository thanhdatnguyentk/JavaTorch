package com.user.nn.losses;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class PairwiseDistance extends Module {
    public float p;
    public float eps;

    public PairwiseDistance(float p, float eps) {
        this.p = p; this.eps = eps;
    }

    public PairwiseDistance() {
        this(2.0f, 1e-6f);
    }

    public Tensor forward(Tensor x1, Tensor x2) {
        return Functional.pairwise_distance(x1, x2, p, eps);
    }
}
