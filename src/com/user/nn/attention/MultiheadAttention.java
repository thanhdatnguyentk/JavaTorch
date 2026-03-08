package com.user.nn.attention;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.Linear;

public class MultiheadAttention extends Module {
    public int embedDim;
    public int numHeads;
    public int headDim;
    public Linear q_proj;
    public Linear k_proj;
    public Linear v_proj;
    public Linear out_proj;
    public float dropout;

    public MultiheadAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.0f);
    }

    public MultiheadAttention(int embedDim, int numHeads, float dropout) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.dropout = dropout;
        if (headDim * numHeads != embedDim)
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");

        this.q_proj = new Linear(embedDim, embedDim, true);
        this.k_proj = new Linear(embedDim, embedDim, true);
        this.v_proj = new Linear(embedDim, embedDim, true);
        this.out_proj = new Linear(embedDim, embedDim, true);
        addModule("q_proj", q_proj);
        addModule("k_proj", k_proj);
        addModule("v_proj", v_proj);
        addModule("out_proj", out_proj);
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, x, x, null);
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor mask) {
        int n = query.shape[0];
        int l = query.shape[1];
        
        Tensor q = q_proj.forward(query);
        Tensor k = k_proj.forward(key);
        Tensor v = v_proj.forward(value);
        
        q = Torch.transpose(q.reshape(n, l, numHeads, headDim), 1, 2);
        k = Torch.transpose(k.reshape(n, key.shape[1], numHeads, headDim), 1, 2);
        v = Torch.transpose(v.reshape(n, value.shape[1], numHeads, headDim), 1, 2);
        
        q = q.reshape(n * numHeads, l, headDim);
        k = k.reshape(n * numHeads, key.shape[1], headDim);
        v = v.reshape(n * numHeads, value.shape[1], headDim);
        
        Tensor kt = Torch.transpose(k, 1, 2);
        Tensor attn = Torch.bmm(q, kt);
        attn = Torch.div(attn, (float)Math.sqrt(headDim));
        
        if (mask != null) attn = Torch.add(attn, mask);
        
        attn = Torch.softmax(attn, -1);
        
        Tensor out = Torch.bmm(attn, v);
        
        out = Torch.transpose(out.reshape(n, numHeads, l, headDim), 1, 2);
        out = out.reshape(n, l, embedDim);
        
        return out_proj.forward(out);
    }
}
