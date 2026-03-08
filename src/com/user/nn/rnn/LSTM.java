package com.user.nn.rnn;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class LSTM extends Module {
    public LSTMCell cell;
    public boolean batchFirst;

    public LSTM(int inputSize, int hiddenSize, boolean bias, boolean batchFirst) {
        this.cell = new LSTMCell(inputSize, hiddenSize, bias);
        this.batchFirst = batchFirst;
        addModule("cell", this.cell);
    }

    @Override
    public Tensor forward(Tensor x) {
        int seqLen = batchFirst ? x.shape[1] : x.shape[0];
        int batch = batchFirst ? x.shape[0] : x.shape[1];
        int inputSize = x.shape[2];
        int hiddenSize = cell.hiddenSize;

        Tensor h = Torch.zeros(batch, hiddenSize).to(x.device);
        Tensor c = Torch.zeros(batch, hiddenSize).to(x.device);

        java.util.List<Tensor> outputs = new java.util.ArrayList<>();
        for (int t = 0; t < seqLen; t++) {
            Tensor xt;
            if (batchFirst) {
                xt = Torch.reshape(Torch.narrow(x, 1, t, 1), batch, inputSize);
            } else {
                xt = Torch.reshape(Torch.narrow(x, 0, t, 1), batch, inputSize);
            }
            Tensor[] nc = cell.forward(xt, h, c);
            h = nc[0];
            c = nc[1];
            outputs.add(h);
        }
        return Torch.stack(outputs, batchFirst ? 1 : 0);
    }
}
