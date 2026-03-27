package com.user.nn.models;

import com.user.nn.core.Module;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.layers.Dropout;
import com.user.nn.layers.Embedding;
import com.user.nn.layers.Linear;
import com.user.nn.rnn.LSTM;

public class MultiTaskLSTMModel extends Module {
    public Embedding embedding;
    public LSTM lstm;
    public Dropout dropout;
    public Linear sentimentHead;
    public Linear topicHead;

    public MultiTaskLSTMModel(int vocabSize, int embedDim, int hiddenDim, int sentimentClasses, int topicClasses) {
        this.embedding = new Embedding(vocabSize, embedDim);
        this.lstm = new LSTM(embedDim, hiddenDim, true, true);
        this.dropout = new Dropout(0.2f);
        this.sentimentHead = new Linear(hiddenDim, sentimentClasses, true);
        this.topicHead = new Linear(hiddenDim, topicClasses, true);

        addModule("embedding", embedding);
        addModule("lstm", lstm);
        addModule("dropout", dropout);
        addModule("sentimentHead", sentimentHead);
        addModule("topicHead", topicHead);
    }

    private Tensor pooledFeatures(Tensor x) {
        Tensor embed = embedding.forward(x);
        Tensor lstmOut = lstm.forward(embed);

        int batch = x.shape[0];
        int seqLen = x.shape[1];
        int hiddenDim = lstm.cell.hiddenSize;

        Tensor lastHidden = Torch.narrow(lstmOut, 1, seqLen - 1, 1);
        lastHidden = Torch.reshape(lastHidden, batch, hiddenDim);
        return dropout.forward(lastHidden);
    }

    public Tensor[] forwardBoth(Tensor x) {
        Tensor shared = pooledFeatures(x);
        Tensor sentimentLogits = sentimentHead.forward(shared);
        Tensor topicLogits = topicHead.forward(shared);
        return new Tensor[] { sentimentLogits, topicLogits };
    }

    @Override
    public Tensor forward(Tensor x) {
        return forwardBoth(x)[0];
    }
}
