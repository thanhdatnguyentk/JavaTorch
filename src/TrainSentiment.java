import com.user.nn.*;
import java.util.*;

public class TrainSentiment {
    public static void main(String[] args) throws Exception {
        nn lib = new nn();
        
        System.out.println("Loading Movie Review dataset...");
        List<MovieCommentLoader.Entry> allData = MovieCommentLoader.load();
        System.out.println("Total reviews: " + allData.size());
        
        // Shuffle and split data
        Collections.shuffle(allData, new Random(42L));
        int total = allData.size();
        int trainSize = (int) (total * 0.8);
        List<MovieCommentLoader.Entry> trainEntries = allData.subList(0, trainSize);
        List<MovieCommentLoader.Entry> testEntries = allData.subList(trainSize, total);
        
        data.Vocabulary vocab = new data.Vocabulary();
        data.BasicTokenizer tokenizer = new data.BasicTokenizer();
        
        // Build vocabulary from train data only
        for (MovieCommentLoader.Entry e : trainEntries) {
            for (String t : tokenizer.tokenize(e.text)) {
                vocab.addWord(t);
            }
        }
        System.out.println("Vocabulary size: " + vocab.size());
        
        int maxLen = 20; // 20 words per review
        int batchSize = 16;
        
        // Train on a larger subset now that Autograd is O(N)
        int demoTrainSize = Math.min(trainEntries.size(), 5000); 
        trainEntries = trainEntries.subList(0, demoTrainSize);
        System.out.println("Using a subset of " + demoTrainSize + " training samples.");
        
        // Model params
        int vocabSize = vocab.size();
        int embedDim = 32;
        int hiddenDim = 64;
        int outputDim = 2;
        
        SentimentModel model = new SentimentModel(lib, vocabSize, embedDim, hiddenDim, outputDim);
        for (nn.Parameter p : model.parameters()) p.getTensor().requires_grad = true;
        
        float lr = 0.001f;
        optim.Adam optimizer = new optim.Adam(model.parameters(), lr);
        
        int epochs = 3;
        System.out.println("\nTraining on " + trainEntries.size() + " samples...");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0f;
            int correct = 0;
            int numBatches = (trainEntries.size() + batchSize - 1) / batchSize;
            
            for (int b = 0; b < numBatches; b++) {
                int start = b * batchSize;
                int end = Math.min(start + batchSize, trainEntries.size());
                int currentBs = end - start;
                
                // Prepare batch
                float[] xData = new float[currentBs * maxLen];
                int[] yLabels = new int[currentBs];
                
                for (int i = 0; i < currentBs; i++) {
                    MovieCommentLoader.Entry entry = trainEntries.get(start + i);
                    List<String> tokens = tokenizer.tokenize(entry.text);
                    for (int j = 0; j < maxLen; j++) {
                        if (j < tokens.size()) xData[i * maxLen + j] = vocab.getId(tokens.get(j));
                        else xData[i * maxLen + j] = 0; // <PAD> (ID 0)
                    }
                    yLabels[i] = entry.label;
                }
                
                Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
                
                optimizer.zero_grad();
                Tensor logits = model.forward(xBatch);
                Tensor loss = nn.F.cross_entropy_tensor(logits, yLabels);
                
                loss.backward();
                optimizer.step();
                
                totalLoss += loss.data[0];
                for (int i = 0; i < currentBs; i++) {
                    int pred = logits.data[i * 2] > logits.data[i * 2 + 1] ? 0 : 1;
                    if (pred == yLabels[i]) correct++;
                }

                if ((b + 1) % 1 == 0) {
                    System.out.printf("  Epoch %d Batch %d/%d - loss: %.4f%n", epoch + 1, b + 1, numBatches, loss.data[0]);
                }
            }
            
            float avgLoss = totalLoss / numBatches;
            float trainAcc = (float) correct / trainEntries.size();
            
            // Evaluation on test set
            float testAcc = evaluate(model, testEntries, vocab, tokenizer, maxLen);
            
            System.out.printf("Epoch %d/%d - loss: %.4f - train_acc: %.2f%% - test_acc: %.2f%%%n",
                epoch + 1, epochs, avgLoss, trainAcc * 100, testAcc * 100);
        }
    }

    private static float evaluate(SentimentModel model, List<MovieCommentLoader.Entry> data, 
                                 data.Vocabulary vocab, data.BasicTokenizer tokenizer, int maxLen) {
        int correct = 0;
        int testBatchSize = 64;
        
        Torch.set_grad_enabled(false);
        for (int i = 0; i < data.size(); i += testBatchSize) {
            int end = Math.min(i + testBatchSize, data.size());
            int bs = end - i;
            float[] xData = new float[bs * maxLen];
            int[] yLabels = new int[bs];
            
            for (int j = 0; j < bs; j++) {
                MovieCommentLoader.Entry e = data.get(i + j);
                List<String> tokens = tokenizer.tokenize(e.text);
                for (int k = 0; k < maxLen; k++) {
                   if (k < tokens.size()) xData[j * maxLen + k] = vocab.getId(tokens.get(k));
                   else xData[j * maxLen + k] = 0;
                }
                yLabels[j] = e.label;
            }
            Tensor xBatch = Torch.tensor(xData, bs, maxLen);
            Tensor out = model.forward(xBatch);
            for (int j = 0; j < bs; j++) {
                int pred = out.data[j * 2] > out.data[j * 2 + 1] ? 0 : 1;
                if (pred == yLabels[j]) correct++;
            }
        }
        Torch.set_grad_enabled(true);
        return (float) correct / data.size();
    }
}
