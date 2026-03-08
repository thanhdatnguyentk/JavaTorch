package com.user.nn.rnn;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class LSTMCell extends Module {
    public int inputSize;
    public int hiddenSize;
    public Parameter weight_ih;
    public Parameter weight_hh;
    public Parameter bias_ih;
    public Parameter bias_hh;

    public LSTMCell(int inputSize, int hiddenSize, boolean bias) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        float k = (float) Math.sqrt(1.0 / hiddenSize);
        NN.Mat w_ih = NN.mat_alloc(inputSize, 4 * hiddenSize);
        NN.mat_rand(w_ih, -k, k);
        this.weight_ih = new Parameter(w_ih);
        addParameter("weight_ih", this.weight_ih);

        NN.Mat w_hh = NN.mat_alloc(hiddenSize, 4 * hiddenSize);
        NN.mat_rand(w_hh, -k, k);
        this.weight_hh = new Parameter(w_hh);
        addParameter("weight_hh", this.weight_hh);

        if (bias) {
            NN.Mat b_ih = NN.mat_alloc(1, 4 * hiddenSize);
            NN.mat_fill(b_ih, 0f);
            this.bias_ih = new Parameter(b_ih);
            addParameter("bias_ih", this.bias_ih);

            NN.Mat b_hh = NN.mat_alloc(1, 4 * hiddenSize);
            NN.mat_fill(b_hh, 0f);
            this.bias_hh = new Parameter(b_hh);
            addParameter("bias_hh", this.bias_hh);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("LSTMCell requires (input, hidden, cell). Use forward(x, h, c).");
    }

    public Tensor[] forward(Tensor x, Tensor h, Tensor c) {
        Tensor x_w = Torch.matmul(x, weight_ih.getTensor());
        if (bias_ih != null) x_w = Torch.add(x_w, bias_ih.getTensor());
        Tensor h_w = Torch.matmul(h, weight_hh.getTensor());
        if (bias_hh != null) h_w = Torch.add(h_w, bias_hh.getTensor());
        Tensor gates = Torch.add(x_w, h_w);

        java.util.List<Tensor> splitGates = Torch.chunk(gates, 4, 1);
        Tensor i_t = Torch.sigmoid(splitGates.get(0));
        Tensor f_t = Torch.sigmoid(splitGates.get(1));
        Tensor g_t = Torch.tanh(splitGates.get(2));
        Tensor o_t = Torch.sigmoid(splitGates.get(3));

        Tensor c_next = Torch.add(Torch.mul(f_t, c), Torch.mul(i_t, g_t));
        Tensor h_next = Torch.mul(o_t, Torch.tanh(c_next));

        return new Tensor[] { h_next, c_next };
    }
}
