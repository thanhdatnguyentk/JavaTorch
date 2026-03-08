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
        indices.toCPU();
        Tensor w = weight.getTensor();
        w.toCPU();
        int[] idxShape = indices.shape;
        int numIdx = indices.numel();
        int[] outShape = new int[idxShape.length + 1];
        System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
        outShape[idxShape.length] = embeddingDim;

        Tensor out = new Tensor(outShape);
        if (w.isGPU() || indices.isGPU()) out.toGPU();

        for (int i = 0; i < numIdx; i++) {
            int idx = (int) indices.data[i];
            if (idx < 0 || idx >= numEmbeddings) {
                throw new IndexOutOfBoundsException("Embedding index out of range: " + idx);
            }
            System.arraycopy(w.data, idx * embeddingDim, out.data, i * embeddingDim, embeddingDim);
        }

        if (Torch.is_grad_enabled() && w.requires_grad) {
            out.requires_grad = true;
            out.grad_fn = new Tensor.GradFn(weight.getTensor()) {
                public void apply(Tensor outGrad) {
                    Tensor gw = new Tensor(w.shape);
                    for (int i = 0; i < numIdx; i++) {
                        int idx = (int) indices.data[i];
                        for (int d = 0; d < embeddingDim; d++) {
                            gw.data[idx * embeddingDim + d] += outGrad.data[i * embeddingDim + d];
                        }
                    }
                    w.backwardStep(gw);
                }
            };
        }
        return out;
    }
}
