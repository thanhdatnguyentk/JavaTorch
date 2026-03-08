package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.models.*;
import com.user.nn.metrics.*;
import java.util.*;

public class TrainSentiment {
    public static void main(String[] args) throws Exception {
        MixedPrecision.enable(); // Opt-in to Tensor Cores (FP16)
        
        NN lib = new NN();
        
        System.out.println("Loading Movie Review dataset...");
        List<MovieCommentLoader.Entry> allData = MovieCommentLoader.load();
        System.out.println("Total reviews: " + allData.size());
        
        Collections.shuffle(allData, new Random(42L));
        int total = allData.size();
        int trainSize = (int) (total * 0.8);
        List<MovieCommentLoader.Entry> trainEntries = allData.subList(0, trainSize);
        List<MovieCommentLoader.Entry> testEntries = allData.subList(trainSize, total);
        
        Data.Vocabulary vocab = new Data.Vocabulary();
        Data.BasicTokenizer tokenizer = new Data.BasicTokenizer();
        
        for (MovieCommentLoader.Entry e : trainEntries) {
            for (String t : tokenizer.tokenize(e.text)) vocab.addWord(t);
        }
        System.out.println("Vocabulary size: " + vocab.size());
        
        int maxLen = 20; 
        int batchSize = 16;
        
        int vocabSize = vocab.size();
        int embedDim = 32;
        int hiddenDim = 64;
        int outputDim = 2;
        
        SentimentModel model = new SentimentModel(lib, vocabSize, embedDim, hiddenDim, outputDim);
        for (NN.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            t.requires_grad = true;
        }

        // Initialize GPU Memory Pool based on model size
        GpuMemoryPool.autoInit(model);

        // Move model to GPU
        model.toGPU();
        
        float lr = 0.001f;
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);
        
        Accuracy accMetric = new Accuracy();
        int epochs = 10;
        System.out.println("\nTraining on " + trainEntries.size() + " samples...");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0f;
            int numBatches = (trainEntries.size() + batchSize - 1) / batchSize;
            accMetric.reset();
            model.train();

            for (int b = 0; b < numBatches; b++) {
                try (MemoryScope scope = new MemoryScope()) {
                    int start = b * batchSize;
                    int end = Math.min(start + batchSize, trainEntries.size());
                    int currentBs = end - start;
                    
                    float[] xData = new float[currentBs * maxLen];
                    for (int i = 0; i < currentBs; i++) {
                        MovieCommentLoader.Entry entry = trainEntries.get(start + i);
                        List<String> tokens = tokenizer.tokenize(entry.text);
                        for (int j = 0; j < maxLen; j++) {
                            if (j < tokens.size()) xData[i * maxLen + j] = vocab.getId(tokens.get(j));
                            else xData[i * maxLen + j] = 0;
                        }
                    }
                    int[] yLabels = new int[currentBs];
                    for (int i = 0; i < currentBs; i++) yLabels[i] = trainEntries.get(start + i).label;

                    Tensor xBatch = Torch.tensor(xData, currentBs, maxLen);
                    xBatch.toGPU();
                    
                    optimizer.zero_grad();
                    Tensor logits = model.forward(xBatch);
                    Tensor loss = NN.F.cross_entropy_tensor(logits, yLabels);
                    
                    loss.backward();
                    optimizer.step();
                    
                    totalLoss += loss.data[0];
                    accMetric.update(logits, yLabels);

                    if ((b + 1) % 100 == 0) {
                        System.out.printf("  Batch %d/%d - loss: %.4f%n", b + 1, numBatches, loss.data[0]);
                    }
                }
            }
            
            float avgLoss = totalLoss / numBatches;
            float trainAcc = accMetric.compute();
            
            Data.Dataset testDataset = new Data.Dataset() {
                @Override
                public int len() { return testEntries.size(); }

                @Override
                public Tensor[] get(int index) {
                    MovieCommentLoader.Entry entry = testEntries.get(index);
                    List<String> tokens = tokenizer.tokenize(entry.text);
                    float[] xData = new float[maxLen];
                    for (int j = 0; j < maxLen; j++) {
                        if (j < tokens.size()) xData[j] = vocab.getId(tokens.get(j));
                        else xData[j] = 0;
                    }
                    Tensor x = Torch.tensor(xData, maxLen);
                    Tensor y = Torch.tensor(new float[] { entry.label }, 1);
                    return new Tensor[] { x, y };
                }
            };
            Data.DataLoader testLoader = new Data.DataLoader(testDataset, 64, false, 1);
            
            float testAcc = Evaluator.evaluate(model, testLoader, accMetric);
            
            System.out.printf("Epoch %d/%d - loss: %.4f - train_acc: %.2f%% - test_acc: %.2f%%%n",
                epoch + 1, epochs, avgLoss, trainAcc * 100, testAcc * 100);
                
            testLoader.shutdown();
        }
    }

}
