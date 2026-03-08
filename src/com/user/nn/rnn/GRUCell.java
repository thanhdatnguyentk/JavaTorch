package com.user.nn.rnn;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class GRUCell extends Module {
    public int inputSize, hiddenSize;
    public Parameter weight_ih, weight_hh, bias_ih, bias_hh;

    public GRUCell(int inputSize, int hiddenSize, boolean bias) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        float k = (float) Math.sqrt(1.0 / hiddenSize);
        NN.Mat w_ih = NN.mat_alloc(inputSize, 3 * hiddenSize);
        NN.mat_rand(w_ih, -k, k);
        this.weight_ih = new Parameter(w_ih);
        addParameter("weight_ih", this.weight_ih);
        NN.Mat w_hh = NN.mat_alloc(hiddenSize, 3 * hiddenSize);
        NN.mat_rand(w_hh, -k, k);
        this.weight_hh = new Parameter(w_hh);
        addParameter("weight_hh", this.weight_hh);
        if (bias) {
            NN.Mat b_ih = NN.mat_alloc(1, 3 * hiddenSize);
            NN.mat_fill(b_ih, 0f);
            this.bias_ih = new Parameter(b_ih);
            addParameter("bias_ih", this.bias_ih);
            NN.Mat b_hh = NN.mat_alloc(1, 3 * hiddenSize);
            NN.mat_fill(b_hh, 0f);
            this.bias_hh = new Parameter(b_hh);
            addParameter("bias_hh", this.bias_hh);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("Forward(x, h) required");
    }

    public Tensor forward(Tensor x, Tensor h) {
        Tensor x_w = Torch.matmul(x, weight_ih.getTensor());
        if (bias_ih != null) x_w = Torch.add(x_w, bias_ih.getTensor());
        Tensor h_w = Torch.matmul(h, weight_hh.getTensor());
        if (bias_hh != null) h_w = Torch.add(h_w, bias_hh.getTensor());
        java.util.List<Tensor> x_g = Torch.chunk(x_w, 3, 1);
        java.util.List<Tensor> h_g = Torch.chunk(h_w, 3, 1);
        Tensor r = Torch.sigmoid(Torch.add(x_g.get(0), h_g.get(0)));
        Tensor z = Torch.sigmoid(Torch.add(x_g.get(1), h_g.get(1)));
        Tensor n = Torch.tanh(Torch.add(x_g.get(2), Torch.mul(r, h_g.get(2))));
        return Torch.add(Torch.mul(Torch.sub(1.0f, z), n), Torch.mul(z, h));
    }
}
