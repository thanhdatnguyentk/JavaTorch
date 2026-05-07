package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.*;
import com.user.nn.predict.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestPredict {

    static class SimpleClassifier extends Module {
        Linear fc1, fc2;
        int inputDim;

        SimpleClassifier(int inputDim, int numClasses) {
            this.inputDim = inputDim;
            fc1 = new Linear(inputDim, 32, true);
            fc2 = new Linear(32, numClasses, true);
            addModule("fc1", fc1);
            addModule("fc2", fc2);
        }

        @Override
        public Tensor forward(Tensor x) {
            if (x.dim() > 2) {
                int batch = x.shape[0];
                x = x.reshape(batch, inputDim);
            }
            x = fc1.forward(x);
            x = Torch.relu(x);
            return fc2.forward(x);
        }
    }

    private float sumArray(float[] arr) {
        float s = 0;
        for (float v : arr) s += v;
        return s;
    }

    @Test
    void testPredictionResult() {
        float[] probs = {0.1f, 0.7f, 0.2f};
        float[] logits = {-1.0f, 2.0f, 0.5f};
        int[] topKIdx = {1, 2, 0};
        float[] topKProbs = {0.7f, 0.2f, 0.1f};
        String[] topKLabels = {"cat", "dog", "bird"};

        PredictionResult result = new PredictionResult(
            1, 0.7f, probs, logits, "cat", topKIdx, topKProbs, topKLabels);

        assertEquals(1, result.getPredictedClass());
        assertEquals(0.7f, result.getConfidence(), 1e-6f);
        assertEquals("cat", result.getPredictedLabel());
        assertEquals(3, result.getNumClasses());
        assertEquals(1, result.getTopKIndices()[0]);
        assertEquals(0.7f, result.getTopKProbabilities()[0], 1e-6f);
        assertEquals("cat", result.getTopKLabels()[0]);
        assertNotNull(result.toString());
    }

    @Test
    void testPredictorSinglePredict() {
        SimpleClassifier model = new SimpleClassifier(10, 5);
        String[] labels = {"A", "B", "C", "D", "E"};
        Predictor predictor = new Predictor(model, labels).topK(3);

        Tensor input = Torch.rand(new int[]{1, 10});
        PredictionResult result = predictor.predict(input);

        assertNotNull(result);
        assertTrue(result.getPredictedClass() >= 0 && result.getPredictedClass() < 5);
        assertTrue(result.getConfidence() > 0f);
        assertTrue(result.getConfidence() <= 1.0f);
        assertEquals(1.0f, sumArray(result.getProbabilities()), 1e-4f);
        assertEquals(3, result.getTopKIndices().length);
        assertEquals(result.getTopKIndices()[0], result.getPredictedClass());
    }

    @Test
    void testPredictorBatchPredict() {
        SimpleClassifier model = new SimpleClassifier(10, 3);
        Predictor predictor = new Predictor(model).topK(2);

        Tensor input = Torch.rand(new int[]{4, 10});
        PredictionResult[] results = predictor.predictBatch(input);

        assertEquals(4, results.length);
        for (PredictionResult result : results) {
            assertTrue(result.getPredictedClass() >= 0 && result.getPredictedClass() < 3);
            assertEquals(1.0f, sumArray(result.getProbabilities()), 1e-4f);
        }
    }

    @Test
    void testPredictClass() {
        SimpleClassifier model = new SimpleClassifier(10, 4);
        String[] labels = {"cat", "dog", "bird", "fish"};
        Predictor predictor = new Predictor(model, labels);

        Tensor input = Torch.rand(new int[]{1, 10});
        int cls = predictor.predictClass(input);
        String label = predictor.predictLabel(input);

        assertTrue(cls >= 0 && cls < 4);
        assertEquals(labels[cls], label);
    }

    @Test
    void testNoGradDuringPredict() {
        SimpleClassifier model = new SimpleClassifier(10, 3);
        Predictor predictor = new Predictor(model);

        Torch.set_grad_enabled(true);
        assertTrue(Torch.is_grad_enabled());

        Tensor input = Torch.rand(new int[]{1, 10});
        predictor.predict(input);

        assertTrue(Torch.is_grad_enabled(), "Grad should be restored after predict");
    }

    @Test
    void testEvalMode() {
        SimpleClassifier model = new SimpleClassifier(10, 3);
        model.train();
        assertTrue(model.is_training());

        new Predictor(model);
        assertFalse(model.is_training(), "Predictor should set model to eval mode");
    }

    @Test
    void testImagePredictor() {
        SimpleClassifier model = new SimpleClassifier(3 * 4 * 4, 10);
        ImagePredictor predictor = new ImagePredictor(model, 3, 4, 4, ImagePredictor.CIFAR10_LABELS);

        float[] pixels = new float[3 * 4 * 4];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (float) Math.random();

        PredictionResult result = predictor.predictFromPixels(pixels);
        assertNotNull(result);
        assertNotNull(result.getPredictedLabel());

        predictor.normalize(new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.25f, 0.25f, 0.25f});
        assertNotNull(predictor.predictFromPixels(pixels));

        float[][] batchPixels = new float[3][3 * 4 * 4];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < batchPixels[i].length; j++)
                batchPixels[i][j] = (float) Math.random();

        assertEquals(3, predictor.predictFromPixelBatch(batchPixels).length);
    }

    @Test
    void testImagePredictorFactory() {
        SimpleClassifier cifar = new SimpleClassifier(3 * 32 * 32, 10);
        ImagePredictor p1 = ImagePredictor.forCifar10(cifar);
        assertEquals(3, p1.getChannels());
        assertEquals(32, p1.getHeight());
        assertEquals(32, p1.getWidth());
        assertEquals(10, p1.getLabels().length);

        SimpleClassifier mnist = new SimpleClassifier(1 * 28 * 28, 10);
        ImagePredictor p2 = ImagePredictor.forMnist(mnist);
        assertEquals(1, p2.getChannels());
        assertEquals(28, p2.getHeight());

        SimpleClassifier fmnist = new SimpleClassifier(1 * 28 * 28, 10);
        ImagePredictor p3 = ImagePredictor.forFashionMnist(fmnist);
        assertEquals("T-shirt/top", p3.getLabels()[0]);
    }

    @Test
    void testTopKOrdering() {
        SimpleClassifier model = new SimpleClassifier(10, 5);
        Predictor predictor = new Predictor(model).topK(5);

        Tensor input = Torch.rand(new int[]{1, 10});
        PredictionResult result = predictor.predict(input);

        float[] topKProbs = result.getTopKProbabilities();
        for (int i = 1; i < topKProbs.length; i++) {
            assertTrue(topKProbs[i] <= topKProbs[i - 1], "top-K should be sorted descending");
        }
        assertEquals(result.getConfidence(), topKProbs[0], 1e-6f);
    }
}
