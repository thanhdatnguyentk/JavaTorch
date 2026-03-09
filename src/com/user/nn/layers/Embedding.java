package com.user.nn.layers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class Embedding extends Module {
    public int numEmbeddings;
    public int embeddingDim;
    public Parameter weight;

    public Embedding(int numEmbeddings, int embeddingDim) {
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        NN.Mat w = NN.mat_alloc(numEmbeddings, embeddingDim);
        float k = (float) Math.sqrt(1.0 / embeddingDim);
        NN.mat_rand(w, -k, k);
        this.weight = new Parameter(w);
        addParameter("weight", this.weight);
    }

    @Override
    public Tensor forward(Tensor indices) {
        return Torch.embedding(weight.getTensor(), indices);
    }
}
