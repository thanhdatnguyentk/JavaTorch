package com.user.nn.models;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.*;
import com.user.nn.rnn.*;

public class SentimentModel extends Module {
    public Embedding embedding;
    public LSTM lstm1;
    public LSTM lstm2;
    public Linear fc;
    public Dropout dropout;

    public SentimentModel(int vocabSize, int embedDim, int hiddenDim, int outputDim) {
        this.embedding = new Embedding(vocabSize, embedDim);
        this.lstm1 = new LSTM(embedDim, hiddenDim, true, true); // batchFirst=true
        this.lstm2 = new LSTM(hiddenDim, hiddenDim, true, true);
        this.dropout = new Dropout(0.4f);
        this.fc = new Linear(hiddenDim, outputDim, true);
        
        addModule("embedding", embedding);
        addModule("lstm1", lstm1);
        addModule("lstm2", lstm2);
        addModule("dropout", dropout);
        addModule("fc", fc);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: [batch, seq_len]
        Tensor embed = embedding.forward(x); // [batch, seq_len, embedDim]
        Tensor lstm1Out = lstm1.forward(embed); // [batch, seq_len, hiddenDim]
        Tensor lstmOut = lstm2.forward(lstm1Out); // [batch, seq_len, hiddenDim]
        
        int batch = x.shape[0];
        int seqLen = x.shape[1];
        int hiddenDim = lstm2.cell.hiddenSize;
        
        // Take the last time step: [batch, 1, hiddenDim]
        Tensor lastHidden = Torch.narrow(lstmOut, 1, seqLen - 1, 1);
        lastHidden = Torch.reshape(lastHidden, batch, hiddenDim);
        
        // Apply dropout
        lastHidden = dropout.forward(lastHidden);
        
        return fc.forward(lastHidden);
    }
}
