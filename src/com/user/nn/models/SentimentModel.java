package com.user.nn.models;

import com.user.nn.core.*;

public class SentimentModel extends NN.Module {
    public NN.Embedding embedding;
    public NN.LSTM lstm;
    public NN.Linear fc;

    public SentimentModel(NN outer, int vocabSize, int embedDim, int hiddenDim, int outputDim) {
        this.embedding = new NN.Embedding(outer, vocabSize, embedDim);
        this.lstm = new NN.LSTM(outer, embedDim, hiddenDim, true, true); // batchFirst=true
        this.fc = new NN.Linear(outer, hiddenDim, outputDim, true);
        
        addModule("embedding", embedding);
        addModule("lstm", lstm);
        addModule("fc", fc);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: [batch, seq_len]
        Tensor embed = embedding.forward(x); // [batch, seq_len, embedDim]
        Tensor lstmOut = lstm.forward(embed); // [batch, seq_len, hiddenDim]
        
        int batch = x.shape[0];
        int seqLen = x.shape[1];
        int hiddenDim = lstm.cell.hiddenSize;
        
        // Take the last time step: [batch, 1, hiddenDim]
        Tensor lastHidden = Torch.narrow(lstmOut, 1, seqLen - 1, 1);
        // Reshape to [batch, hiddenDim]
        return fc.forward(Torch.reshape(lastHidden, batch, hiddenDim));
    }
}
