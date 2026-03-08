package com.user.nn.rnn;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class RNNCell extends Module {
    public int inputSize;
    public int hiddenSize;
    public Parameter weight_ih;
    public Parameter weight_hh;
    public Parameter bias_ih;
    public Parameter bias_hh;

    public RNNCell(int inputSize, int hiddenSize, boolean bias) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        float k = (float) Math.sqrt(1.0 / hiddenSize);
        NN.Mat w_ih = NN.mat_alloc(inputSize, hiddenSize);
        NN.mat_rand(w_ih, -k, k);
        this.weight_ih = new Parameter(w_ih);
        addParameter("weight_ih", this.weight_ih);

        NN.Mat w_hh = NN.mat_alloc(hiddenSize, hiddenSize);
        NN.mat_rand(w_hh, -k, k);
        this.weight_hh = new Parameter(w_hh);
        addParameter("weight_hh", this.weight_hh);

        if (bias) {
            NN.Mat b_ih = NN.mat_alloc(1, hiddenSize);
            NN.mat_fill(b_ih, 0f);
            this.bias_ih = new Parameter(b_ih);
            addParameter("bias_ih", this.bias_ih);

            NN.Mat b_hh = NN.mat_alloc(1, hiddenSize);
            NN.mat_fill(b_hh, 0f);
            this.bias_hh = new Parameter(b_hh);
            addParameter("bias_hh", this.bias_hh);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("RNNCell requires (input, hidden). Use forward(x, h).");
    }

    public Tensor forward(Tensor x, Tensor h) {
        Tensor x_w = Torch.matmul(x, weight_ih.getTensor());
        if (bias_ih != null) x_w = Torch.add(x_w, bias_ih.getTensor());
        Tensor h_w = Torch.matmul(h, weight_hh.getTensor());
        if (bias_hh != null) h_w = Torch.add(h_w, bias_hh.getTensor());
        return Torch.tanh(Torch.add(x_w, h_w));
    }
}
