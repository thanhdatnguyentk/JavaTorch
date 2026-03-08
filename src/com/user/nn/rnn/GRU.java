package com.user.nn.rnn;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class GRU extends Module {
    public GRUCell cell;
    public boolean batchFirst;

    public GRU(int inputSize, int hiddenSize, boolean bias, boolean batchFirst) {
        this.cell = new GRUCell(inputSize, hiddenSize, bias);
        this.batchFirst = batchFirst;
        addModule("cell", this.cell);
    }

    @Override
    public Tensor forward(Tensor x) {
        int seqLen = batchFirst ? x.shape[1] : x.shape[0];
        int batch = batchFirst ? x.shape[0] : x.shape[1];
        int inputSize = x.shape[2];
        Tensor h = Torch.zeros(batch, cell.hiddenSize).to(x.device);
        java.util.List<Tensor> outputs = new java.util.ArrayList<>();
        for (int t = 0; t < seqLen; t++) {
            Tensor xt = batchFirst ? Torch.reshape(Torch.narrow(x, 1, t, 1), batch, inputSize)
                                 : Torch.reshape(Torch.narrow(x, 0, t, 1), batch, inputSize);
            h = cell.forward(xt, h);
            outputs.add(h);
        }
        return Torch.stack(outputs, batchFirst ? 1 : 0);
    }
}
