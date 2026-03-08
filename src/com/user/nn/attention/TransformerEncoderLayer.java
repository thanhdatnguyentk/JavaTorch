package com.user.nn.attention;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.Linear;
import com.user.nn.norm.LayerNorm;

public class TransformerEncoderLayer extends Module {
    public MultiheadAttention self_attn;
    public Linear linear1, linear2;
    public LayerNorm norm1, norm2;
    public float dropoutP;

    public TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward, float dropout) {
        this.self_attn = new MultiheadAttention(d_model, nhead, dropout);
        this.linear1 = new Linear(d_model, dim_feedforward, true);
        this.linear2 = new Linear(dim_feedforward, d_model, true);
        this.norm1 = new LayerNorm(d_model);
        this.norm2 = new LayerNorm(d_model);
        this.dropoutP = dropout;
        addModule("self_attn", self_attn);
        addModule("linear1", linear1);
        addModule("linear2", linear2);
        addModule("norm1", norm1);
        addModule("norm2", norm2);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor x1 = norm1.forward(x);
        Tensor attn_out = self_attn.forward(x1);
        x = Torch.add(x, attn_out);

        Tensor x2 = norm2.forward(x);
        Tensor mlp_out = linear2.forward(Torch.relu(linear1.forward(x2)));
        x = Torch.add(x, mlp_out);
        return x;
    }
}
