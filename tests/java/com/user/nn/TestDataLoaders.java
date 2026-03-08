package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import java.util.List;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;

/**
 * Tests for DataLoader, BaseDataset, Vocabulary, BasicTokenizer, and MAE metric.
 */
public class TestDataLoaders {
    static int failures = 0;

    public static void main(String[] args) {
        System.out.println("Running TestDataLoaders...");

        testBaseDataset();
        testDataLoaderIteration();
        testDataLoaderShuffle();
        testVocabulary();
        testBasicTokenizer();
        testMAEMetric();
        testMetricTracker();

        if (failures > 0) {
            System.out.println("TestDataLoaders FAILED (" + failures + " failures).");
            System.exit(1);
        }
        System.out.println("TestDataLoaders PASSED.");
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            failures++;
        }
    }

    private static void testBaseDataset() {
        float[][] data = {
            {1f, 2f, 0f},
            {3f, 4f, 1f},
            {5f, 6f, 0f},
            {7f, 8f, 1f}
        };
        Data.BaseDataset ds = new Data.BaseDataset(data);

        check(ds.len() == 4, "BaseDataset len should be 4, got " + ds.len());

        Tensor[] sample = ds.get(0);
        check(sample != null, "BaseDataset.get(0) should not be null");
        check(sample[0].numel() == 3, "sample[0] should have 3 elements");
        check(Math.abs(sample[0].data[0] - 1f) < 1e-6f, "sample[0].data[0] should be 1.0");

        Tensor[] sample3 = ds.get(3);
        check(Math.abs(sample3[0].data[0] - 7f) < 1e-6f, "sample[3].data[0] should be 7.0");

        System.out.println("  BaseDataset OK");
    }

    private static void testDataLoaderIteration() {
        float[][] data = new float[10][3]; // 10 samples, 3 features each
        for (int i = 0; i < 10; i++) {
            data[i] = new float[]{i, i * 2f, i * 3f};
        }
        Data.BaseDataset ds = new Data.BaseDataset(data);

        // batchSize=3, no shuffle, 1 worker
        Data.DataLoader loader = new Data.DataLoader(ds, 3, false, 1);

        int batchCount = 0;
        int totalSamples = 0;
        for (Tensor[] batch : loader) {
            batchCount++;
            totalSamples += batch[0].shape[0];
        }
        loader.shutdown();

        // 10 samples / batch 3 = 4 batches (3+3+3+1)
        check(batchCount == 4, "Expected 4 batches, got " + batchCount);
        check(totalSamples == 10, "Expected 10 total samples, got " + totalSamples);

        System.out.println("  DataLoader iteration OK");
    }

    private static void testDataLoaderShuffle() {
        float[][] data = new float[20][2];
        for (int i = 0; i < 20; i++) {
            data[i] = new float[]{i, 0f};
        }
        Data.BaseDataset ds = new Data.BaseDataset(data);

        // Collect first elements from non-shuffled pass
        Data.DataLoader loaderNoShuffle = new Data.DataLoader(ds, 5, false, 1);
        List<Float> orderedValues = new ArrayList<>();
        for (Tensor[] batch : loaderNoShuffle) {
            batch[0].toCPU();
            for (int i = 0; i < batch[0].shape[0]; i++) {
                orderedValues.add(batch[0].data[i * 2]); // first feature per sample
            }
        }
        loaderNoShuffle.shutdown();

        // Collect from shuffled pass
        Data.DataLoader loaderShuffle = new Data.DataLoader(ds, 5, true, 1);
        List<Float> shuffledValues = new ArrayList<>();
        for (Tensor[] batch : loaderShuffle) {
            batch[0].toCPU();
            for (int i = 0; i < batch[0].shape[0]; i++) {
                shuffledValues.add(batch[0].data[i * 2]);
            }
        }
        loaderShuffle.shutdown();

        // Same count
        check(orderedValues.size() == shuffledValues.size(),
              "Shuffle should produce same count");

        // Shuffled should (very likely) have a different order
        boolean different = false;
        for (int i = 0; i < orderedValues.size(); i++) {
            if (Math.abs(orderedValues.get(i) - shuffledValues.get(i)) > 1e-6f) {
                different = true;
                break;
            }
        }
        check(different, "Shuffle should produce different order (may rarely fail)");

        // Both should contain same set of values
        Set<Float> orderedSet = new HashSet<>(orderedValues);
        Set<Float> shuffledSet = new HashSet<>(shuffledValues);
        check(orderedSet.equals(shuffledSet), "Shuffle should contain same values, just reordered");

        System.out.println("  DataLoader shuffle OK");
    }

    private static void testVocabulary() {
        Data.Vocabulary vocab = new Data.Vocabulary();

        // PAD=0, UNK=1 by default
        check(vocab.getId(vocab.padToken) == 0, "PAD should be ID 0");
        check(vocab.getId(vocab.unkToken) == 1, "UNK should be ID 1");
        check(vocab.size() >= 2, "Vocab should have at least PAD and UNK");

        // Add words
        int helloId = vocab.addWord("hello");
        int worldId = vocab.addWord("world");
        check(helloId >= 2, "hello should get ID >= 2");
        check(worldId == helloId + 1, "world should follow hello");

        // Duplicate add returns same ID
        int helloId2 = vocab.addWord("hello");
        check(helloId == helloId2, "Duplicate addWord should return same ID");

        // Reverse lookup
        check(vocab.getWord(helloId).equals("hello"), "getWord should return 'hello'");
        check(vocab.getWord(worldId).equals("world"), "getWord should return 'world'");

        // Unknown word returns UNK ID
        int unknownId = vocab.getId("nonexistent");
        check(unknownId == 1, "Unknown word should return UNK ID (1)");

        // Out-of-bounds ID returns UNK token
        check(vocab.getWord(-1).equals(vocab.unkToken), "getWord(-1) should return UNK");
        check(vocab.getWord(99999).equals(vocab.unkToken), "getWord(99999) should return UNK");

        System.out.println("  Vocabulary OK");
    }

    private static void testBasicTokenizer() {
        Data.BasicTokenizer tokenizer = new Data.BasicTokenizer();

        List<String> tokens = tokenizer.tokenize("Hello, World! This is a TEST.");
        check(tokens.size() == 6, "Expected 6 tokens, got " + tokens.size() + ": " + tokens);
        check(tokens.get(0).equals("hello"), "First token should be 'hello', got '" + tokens.get(0) + "'");
        check(tokens.get(1).equals("world"), "Second token should be 'world'");
        check(tokens.get(5).equals("test"), "Last token should be 'test'");

        // Numbers preserved
        List<String> numTokens = tokenizer.tokenize("I have 42 apples");
        check(numTokens.contains("42"), "Tokenizer should preserve numbers");

        // Empty string
        List<String> emptyTokens = tokenizer.tokenize("");
        check(emptyTokens.isEmpty(), "Empty string should give 0 tokens");

        // Only punctuation
        List<String> punctTokens = tokenizer.tokenize("...!!!???");
        check(punctTokens.isEmpty(), "Only punctuation should give 0 tokens");

        System.out.println("  BasicTokenizer OK");
    }

    private static void testMAEMetric() {
        MeanAbsoluteError mae = new MeanAbsoluteError();

        Tensor preds = Torch.tensor(new float[]{1f, 2f, 3f}, 3);
        Tensor targets = Torch.tensor(new float[]{1.5f, 2.5f, 3.5f}, 3);

        mae.update(preds, targets);
        // MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        check(Math.abs(mae.compute() - 0.5f) < 1e-5f, "MAE should be 0.5, got " + mae.compute());

        // Second batch
        Tensor preds2 = Torch.tensor(new float[]{0f, 0f}, 2);
        Tensor targets2 = Torch.tensor(new float[]{1f, 2f}, 2);
        mae.update(preds2, targets2);
        // Total: (0.5+0.5+0.5+1.0+2.0) / 5 = 4.5/5 = 0.9
        check(Math.abs(mae.compute() - 0.9f) < 1e-5f, "Cumulative MAE should be 0.9, got " + mae.compute());

        mae.reset();
        check(mae.compute() == 0f, "MAE after reset should be 0");

        System.out.println("  MAE metric OK");
    }

    private static void testMetricTracker() {
        MetricTracker tracker = new MetricTracker();

        // Update a named metric
        tracker.update("loss", 1.0f);
        tracker.update("loss", 2.0f);
        tracker.update("loss", 3.0f);
        check(Math.abs(tracker.getAverage("loss") - 2.0f) < 1e-5f,
              "Average loss should be 2.0, got " + tracker.getAverage("loss"));

        // Unknown metric should return 0, not NPE
        float unknown = tracker.getAverage("nonexistent");
        check(unknown == 0f, "Unknown metric average should be 0, got " + unknown);

        // Reset
        tracker.reset();
        check(tracker.getAverage("loss") == 0f, "After reset, average should be 0");

        System.out.println("  MetricTracker OK");
    }
}
