import com.user.nn.*;

public class SentimentModel extends nn.Module {
    public nn.Embedding embedding;
    public nn.LSTM lstm;
    public nn.Linear fc;

    public SentimentModel(nn outer, int vocabSize, int embedDim, int hiddenDim, int outputDim) {
        this.embedding = new nn.Embedding(outer, vocabSize, embedDim);
        this.lstm = new nn.LSTM(outer, embedDim, hiddenDim, true, true); // batchFirst=true
        this.fc = new nn.Linear(outer, hiddenDim, outputDim, true);
        
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
