package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;


import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.*;
import com.user.nn.models.*;
import com.user.nn.metrics.*;
import java.util.*;

public class TrainSentiment {
    public static void main(String[] args) throws Exception {
        MixedPrecision.enable(); // Opt-in to Tensor Cores (FP16)
        
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
        
        SentimentModel model = new SentimentModel(vocabSize, embedDim, hiddenDim, outputDim);
        for (Parameter p : model.parameters()) {
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
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("nlp");
        dashboard.setModelInfo("LSTM-Sentiment", epochs);
        try {
            com.user.nn.predict.TextPredictor predictor = com.user.nn.predict.TextPredictor.forSentiment(model, vocab, maxLen);
            DashboardIntegrationHelper.setupTextPredictorHandler(dashboard, "sentiment", predictor);
        } catch(Exception e) {}

        
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
                    Tensor loss = Functional.cross_entropy_tensor(logits, yLabels);
                    
                    loss.backward();
                    optimizer.step();
                    
                    totalLoss += loss.data[0];
                    accMetric.update(logits, yLabels);

                    if ((b + 1) % 100 == 0) {
                        System.out.printf("  Batch %d/%d - loss: %.4f%n", b + 1, numBatches, loss.data[0]);
                        
                        // Real-time Dashboard Visualization
                        try {
                            // Pick the first sample in the batch for visualization
                            MovieCommentLoader.Entry sampleEntry = trainEntries.get(start);
                            String text = sampleEntry.text;
                            
                            // Get model prediction for this specific sample
                            // logits has shape [currentBs, 2]
                            float logitNeg = logits.data[0];
                            float logitPos = logits.data[1];
                            float probPos = (float) (Math.exp(logitPos) / (Math.exp(logitNeg) + Math.exp(logitPos)));
                            
                            Map<String, Float> dashMetrics = new HashMap<>();
                            dashMetrics.put("loss", loss.data[0]);
                            dashMetrics.put("acc", accMetric.compute());
                            
                            // Mocking detailed NLP metrics (F1/Precision/Recall/Token Weights) for demo
                            Map<String, Float> f1 = Map.of("Positive", accMetric.compute() + 0.05f, "Negative", accMetric.compute() - 0.02f);
                            Map<String, Float> precision = Map.of("Positive", accMetric.compute() + 0.02f, "Negative", accMetric.compute() - 0.05f);
                            Map<String, Float> recall = Map.of("Positive", accMetric.compute() - 0.01f, "Negative", accMetric.compute() + 0.03f);
                            
                            Map<String, Float> tokenW = new HashMap<>();
                            List<String> tokens = tokenizer.tokenize(text);
                            for (String tok : tokens) tokenW.put(tok, (float) Math.random()); // simple random attention mock
                            
                            DashboardIntegrationHelper.broadcastNLPDetailed(
                                dashboard, epoch + 1, dashMetrics, text, 
                                probPos > 0.5 ? "POSITIVE" : "NEGATIVE", 
                                Math.max(probPos, 1 - probPos),
                                f1, precision, recall, tokenW
                            );
                        } catch (Exception dashEx) {}
                    }
                    while (dashboard.isTrainingPaused()) {
                        try { Thread.sleep(200); } catch (InterruptedException ie) { break; }
                    }
                }
            }
            
            float avgLoss = totalLoss / numBatches;
            float trainAcc = accMetric.compute();
            dashboard.setCurrentEpoch(epoch + 1);
            
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
        
            try {
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("loss", avgLoss);
                metrics.put("train_acc", trainAcc);
                metrics.put("test_acc", testAcc);
                history.record(epoch + 1, metrics);
                dashboard.broadcastMetrics(epoch + 1, metrics);
            } catch (Exception dashEx) {}
}

        // ============================================================
        //  PREDICTION - Sử dụng thư viện predict
        // ============================================================
        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.println("║       PREDICTION WITH TRAINED MODEL      ║");
        System.out.println("╚══════════════════════════════════════════╝\n");

        model.save("sentiment_model.bin");

        com.user.nn.predict.TextPredictor textPredictor = 
            com.user.nn.predict.TextPredictor.forSentiment(model, vocab, maxLen);
        textPredictor.verbose(true);

        // Predict trên các câu mẫu
        String[] sampleTexts = {
            "This movie is absolutely amazing and wonderful!",
            "Terrible film, waste of time and money.",
            "The acting was decent but the plot was boring.",
            "One of the best movies I have ever seen!",
            "I would not recommend this to anyone."
        };

        System.out.println(">>> Predicting sentiment on sample texts...\n");
        for (String text : sampleTexts) {
            com.user.nn.predict.PredictionResult result = textPredictor.predictText(text);
            System.out.printf("  \"%s\"%n", text);
            System.out.printf("    → %s (confidence: %.4f)%n%n",
                result.getPredictedLabel(), result.getConfidence());
        }

        System.out.println("Training Complete!");
    }

}
