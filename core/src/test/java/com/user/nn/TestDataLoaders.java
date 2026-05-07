package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.dataloaders.*;
import com.user.nn.metrics.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;

public class TestDataLoaders {

    @Test
    void testBaseDataset() {
        float[][] data = {
            {1f, 2f, 0f},
            {3f, 4f, 1f},
            {5f, 6f, 0f},
            {7f, 8f, 1f}
        };
        Data.BaseDataset ds = new Data.BaseDataset(data);

        assertEquals(4, ds.len(), "BaseDataset len mismatch");

        Tensor[] sample = ds.get(0);
        assertNotNull(sample, "BaseDataset.get(0) should not be null");
        assertEquals(3, sample[0].numel());
        assertEquals(1.0f, sample[0].data[0], 1e-6f);

        Tensor[] sample3 = ds.get(3);
        assertEquals(7.0f, sample3[0].data[0], 1e-6f);
    }

    @Test
    void testDataLoaderIteration() {
        float[][] data = new float[10][3];
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

        assertEquals(4, batchCount, "Expected 4 batches (3+3+3+1)");
        assertEquals(10, totalSamples, "Total samples mismatch");
    }

    @Test
    void testDataLoaderShuffle() {
        float[][] data = new float[20][2];
        for (int i = 0; i < 20; i++) {
            data[i] = new float[]{i, 0f};
        }
        Data.BaseDataset ds = new Data.BaseDataset(data);

        // Collect from non-shuffled pass
        Data.DataLoader loaderNoShuffle = new Data.DataLoader(ds, 5, false, 1);
        List<Float> orderedValues = new ArrayList<>();
        for (Tensor[] batch : loaderNoShuffle) {
            batch[0].toCPU();
            for (int i = 0; i < batch[0].shape[0]; i++) {
                orderedValues.add(batch[0].data[i * 2]);
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

        assertEquals(orderedValues.size(), shuffledValues.size(), "Size mismatch after shuffle");

        boolean different = false;
        for (int i = 0; i < orderedValues.size(); i++) {
            if (Math.abs(orderedValues.get(i) - shuffledValues.get(i)) > 1e-6f) {
                different = true;
                break;
            }
        }
        assertTrue(different, "Shuffle should produce a different order (highly likely)");

        Set<Float> orderedSet = new HashSet<>(orderedValues);
        Set<Float> shuffledSet = new HashSet<>(shuffledValues);
        assertEquals(orderedSet, shuffledSet, "Sets mismatch after shuffle");
    }

    @Test
    void testVocabulary() {
        Data.Vocabulary vocab = new Data.Vocabulary();

        assertEquals(0, vocab.getId(vocab.padToken));
        assertEquals(1, vocab.getId(vocab.unkToken));
        assertTrue(vocab.size() >= 2);

        int helloId = vocab.addWord("hello");
        int worldId = vocab.addWord("world");
        assertTrue(helloId >= 2);
        assertEquals(helloId + 1, worldId);

        assertEquals(helloId, vocab.addWord("hello"), "Duplicate add mismatch");

        assertEquals("hello", vocab.getWord(helloId));
        assertEquals("world", vocab.getWord(worldId));
        assertEquals(1, vocab.getId("nonexistent"), "Unknown word should return UNK ID");
        assertEquals(vocab.unkToken, vocab.getWord(-1));
    }

    @Test
    void testBasicTokenizer() {
        Data.BasicTokenizer tokenizer = new Data.BasicTokenizer();
        List<String> tokens = tokenizer.tokenize("Hello, World! This is a TEST.");
        
        assertEquals(6, tokens.size());
        assertEquals("hello", tokens.get(0));
        assertEquals("world", tokens.get(1));
        assertEquals("test", tokens.get(5));

        assertTrue(tokenizer.tokenize("I have 42 apples").contains("42"));
        assertTrue(tokenizer.tokenize("").isEmpty());
        assertTrue(tokenizer.tokenize("...!!!???").isEmpty());
    }

    @Test
    void testMAEMetric() {
        MeanAbsoluteError mae = new MeanAbsoluteError();
        Tensor preds = Torch.tensor(new float[]{1f, 2f, 3f}, 3);
        Tensor targets = Torch.tensor(new float[]{1.5f, 2.5f, 3.5f}, 3);

        mae.update(preds, targets);
        assertEquals(0.5f, mae.compute(), 1e-5f);

        mae.update(Torch.tensor(new float[]{0f, 0f}, 2), Torch.tensor(new float[]{1f, 2f}, 2));
        assertEquals(0.9f, mae.compute(), 1e-5f);

        mae.reset();
        assertEquals(0f, mae.compute());
    }

    @Test
    void testMetricTracker() {
        MetricTracker tracker = new MetricTracker();
        tracker.update("loss", 1.0f);
        tracker.update("loss", 2.0f);
        tracker.update("loss", 3.0f);
        
        assertEquals(2.0f, tracker.getAverage("loss"), 1e-5f);
        assertEquals(0f, tracker.getAverage("nonexistent"));

        tracker.reset();
        assertEquals(0f, tracker.getAverage("loss"));
    }
}
