package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.*;
import com.user.nn.predict.*;

/**
 * Unit tests for the predict library.
 */
public class TestPredict {

    static int pass = 0, fail = 0;

    static void check(String name, boolean cond) {
        if (cond) {
            System.out.println("  PASS: " + name);
            pass++;
        } else {
            System.out.println("  FAIL: " + name);
            fail++;
        }
    }

    /** Simple 2-layer classification model for testing. */
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
            // Flatten if input is more than 2D (e.g., [N, C, H, W] -> [N, C*H*W])
            if (x.dim() > 2) {
                int batch = x.shape[0];
                x = x.reshape(batch, inputDim);
            }
            x = fc1.forward(x);
            x = Torch.relu(x);
            return fc2.forward(x);
        }
    }

    // ============================= TESTS =============================

    static void testPredictionResult() {
        System.out.println("\n=== Test PredictionResult ===");

        float[] probs = {0.1f, 0.7f, 0.2f};
        float[] logits = {-1.0f, 2.0f, 0.5f};
        int[] topKIdx = {1, 2, 0};
        float[] topKProbs = {0.7f, 0.2f, 0.1f};
        String[] topKLabels = {"cat", "dog", "bird"};

        PredictionResult result = new PredictionResult(
            1, 0.7f, probs, logits, "cat", topKIdx, topKProbs, topKLabels);

        check("predictedClass", result.getPredictedClass() == 1);
        check("confidence", Math.abs(result.getConfidence() - 0.7f) < 1e-6);
        check("predictedLabel", "cat".equals(result.getPredictedLabel()));
        check("numClasses", result.getNumClasses() == 3);
        check("topKIndices[0]", result.getTopKIndices()[0] == 1);
        check("topKProbabilities[0]", Math.abs(result.getTopKProbabilities()[0] - 0.7f) < 1e-6);
        check("topKLabels[0]", "cat".equals(result.getTopKLabels()[0]));
        check("toString not null", result.toString() != null && !result.toString().isEmpty());
    }

    static void testPredictorSinglePredict() {
        System.out.println("\n=== Test Predictor - Single Predict ===");

        SimpleClassifier model = new SimpleClassifier(10, 5);
        String[] labels = {"A", "B", "C", "D", "E"};
        Predictor predictor = new Predictor(model, labels).topK(3);

        Tensor input = Torch.rand(new int[]{1, 10});
        PredictionResult result = predictor.predict(input);

        check("result not null", result != null);
        check("predictedClass in range", result.getPredictedClass() >= 0 && result.getPredictedClass() < 5);
        check("confidence > 0", result.getConfidence() > 0f);
        check("confidence <= 1", result.getConfidence() <= 1.0f);
        check("probabilities sum ~1", Math.abs(sumArray(result.getProbabilities()) - 1.0f) < 1e-4);
        check("topK length == 3", result.getTopKIndices().length == 3);
        check("label is valid", result.getPredictedLabel() != null);
        check("topK[0] == predictedClass", result.getTopKIndices()[0] == result.getPredictedClass());
    }

    static void testPredictorBatchPredict() {
        System.out.println("\n=== Test Predictor - Batch Predict ===");

        SimpleClassifier model = new SimpleClassifier(10, 3);
        Predictor predictor = new Predictor(model).topK(2);

        Tensor input = Torch.rand(new int[]{4, 10});
        PredictionResult[] results = predictor.predictBatch(input);

        check("batch size 4", results.length == 4);
        for (int i = 0; i < results.length; i++) {
            check("result[" + i + "] class in range",
                results[i].getPredictedClass() >= 0 && results[i].getPredictedClass() < 3);
            check("result[" + i + "] probs sum ~1",
                Math.abs(sumArray(results[i].getProbabilities()) - 1.0f) < 1e-4);
        }
    }

    static void testPredictClass() {
        System.out.println("\n=== Test Predictor - predictClass / predictLabel ===");

        SimpleClassifier model = new SimpleClassifier(10, 4);
        String[] labels = {"cat", "dog", "bird", "fish"};
        Predictor predictor = new Predictor(model, labels);

        Tensor input = Torch.rand(new int[]{1, 10});
        int cls = predictor.predictClass(input);
        String label = predictor.predictLabel(input);

        check("class in range", cls >= 0 && cls < 4);
        check("label matches", labels[cls].equals(label));
    }

    static void testNoGradDuringPredict() {
        System.out.println("\n=== Test Predictor - No Grad During Predict ===");

        SimpleClassifier model = new SimpleClassifier(10, 3);
        Predictor predictor = new Predictor(model);

        // Grad should be enabled before
        Torch.set_grad_enabled(true);
        check("grad enabled before", Torch.is_grad_enabled());

        Tensor input = Torch.rand(new int[]{1, 10});
        predictor.predict(input);

        // Grad should still be enabled after (restored)
        check("grad restored after predict", Torch.is_grad_enabled());
    }

    static void testEvalMode() {
        System.out.println("\n=== Test Predictor - Eval Mode ===");

        SimpleClassifier model = new SimpleClassifier(10, 3);
        model.train(); // Manually set to training
        check("model is training before", model.is_training());

        Predictor predictor = new Predictor(model);
        // Predictor constructor should set to eval
        check("model is eval after Predictor()", !model.is_training());
    }

    static void testImagePredictor() {
        System.out.println("\n=== Test ImagePredictor ===");

        SimpleClassifier model = new SimpleClassifier(3 * 4 * 4, 10);
        ImagePredictor predictor = new ImagePredictor(model, 3, 4, 4, ImagePredictor.CIFAR10_LABELS);

        // Test predict from pixels
        float[] pixels = new float[3 * 4 * 4];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (float) Math.random();

        PredictionResult result = predictor.predictFromPixels(pixels);
        check("image result not null", result != null);
        check("CIFAR label valid", result.getPredictedLabel() != null);

        // Test normalization
        predictor.normalize(
            new float[]{0.5f, 0.5f, 0.5f},
            new float[]{0.25f, 0.25f, 0.25f}
        );
        PredictionResult normResult = predictor.predictFromPixels(pixels);
        check("normalized result not null", normResult != null);

        // Test batch
        float[][] batchPixels = new float[3][3 * 4 * 4];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < batchPixels[i].length; j++)
                batchPixels[i][j] = (float) Math.random();

        PredictionResult[] batchResults = predictor.predictFromPixelBatch(batchPixels);
        check("batch size 3", batchResults.length == 3);
    }

    static void testImagePredictorFactory() {
        System.out.println("\n=== Test ImagePredictor - Factory Methods ===");

        // Just test that factories set correct values (can't run model since input dims differ)
        SimpleClassifier cifar = new SimpleClassifier(3 * 32 * 32, 10);
        ImagePredictor p1 = ImagePredictor.forCifar10(cifar);
        check("CIFAR-10 channels", p1.getChannels() == 3);
        check("CIFAR-10 height", p1.getHeight() == 32);
        check("CIFAR-10 width", p1.getWidth() == 32);
        check("CIFAR-10 labels count", p1.getLabels().length == 10);

        SimpleClassifier mnist = new SimpleClassifier(1 * 28 * 28, 10);
        ImagePredictor p2 = ImagePredictor.forMnist(mnist);
        check("MNIST channels", p2.getChannels() == 1);
        check("MNIST height", p2.getHeight() == 28);
        check("MNIST labels count", p2.getLabels().length == 10);

        SimpleClassifier fmnist = new SimpleClassifier(1 * 28 * 28, 10);
        ImagePredictor p3 = ImagePredictor.forFashionMnist(fmnist);
        check("Fashion-MNIST labels[0]", "T-shirt/top".equals(p3.getLabels()[0]));
    }

    static void testImagePredictorRgb() {
        System.out.println("\n=== Test ImagePredictor - RGB int[] Input ===");

        SimpleClassifier model = new SimpleClassifier(3 * 4 * 4, 5);
        ImagePredictor predictor = new ImagePredictor(model, 3, 4, 4);

        int[] rgbPixels = new int[3 * 4 * 4];
        for (int i = 0; i < rgbPixels.length; i++) rgbPixels[i] = (int)(Math.random() * 255);

        PredictionResult result = predictor.predictFromRgb(rgbPixels);
        check("RGB result not null", result != null);
        check("RGB class in range", result.getPredictedClass() >= 0 && result.getPredictedClass() < 5);
    }

    static void testPredictionPipeline() {
        System.out.println("\n=== Test PredictionPipeline ===");

        SimpleClassifier model = new SimpleClassifier(10, 3);
        String[] labels = {"X", "Y", "Z"};

        try {
            Predictor predictor = PredictionPipeline
                .create(model)
                .labels(labels)
                .topK(2)
                .verbose(false)
                .build();

            check("pipeline build success", predictor != null);

            Tensor input = Torch.rand(new int[]{1, 10});
            PredictionResult result = predictor.predict(input);
            check("pipeline predict success", result != null);
            check("pipeline label valid", 
                "X".equals(result.getPredictedLabel()) || 
                "Y".equals(result.getPredictedLabel()) || 
                "Z".equals(result.getPredictedLabel()));

        } catch (Exception e) {
            check("pipeline no exception", false);
            e.printStackTrace();
        }
    }

    static void testPredictionPipelineImageMode() {
        System.out.println("\n=== Test PredictionPipeline - Image Mode ===");

        SimpleClassifier model = new SimpleClassifier(3 * 4 * 4, 5);

        try {
            Predictor predictor = PredictionPipeline
                .create(model)
                .imageSize(3, 4, 4)
                .normalize(new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.25f, 0.25f, 0.25f})
                .topK(3)
                .build();

            check("image pipeline is ImagePredictor", predictor instanceof ImagePredictor);

            float[] pixels = new float[3 * 4 * 4];
            for (int i = 0; i < pixels.length; i++) pixels[i] = (float) Math.random();

            PredictionResult result = ((ImagePredictor) predictor).predictFromPixels(pixels);
            check("image pipeline predict success", result != null);

        } catch (Exception e) {
            check("image pipeline no exception", false);
            e.printStackTrace();
        }
    }

    static void testTopKValues() {
        System.out.println("\n=== Test Top-K Ordering ===");

        SimpleClassifier model = new SimpleClassifier(10, 5);
        Predictor predictor = new Predictor(model).topK(5);

        Tensor input = Torch.rand(new int[]{1, 10});
        PredictionResult result = predictor.predict(input);

        float[] topKProbs = result.getTopKProbabilities();
        boolean sorted = true;
        for (int i = 1; i < topKProbs.length; i++) {
            if (topKProbs[i] > topKProbs[i - 1]) {
                sorted = false;
                break;
            }
        }
        check("top-K is sorted descending", sorted);
        check("top-K[0] == confidence", Math.abs(topKProbs[0] - result.getConfidence()) < 1e-6);
    }

    static void testSoftmaxCorrectness() {
        System.out.println("\n=== Test Softmax Correctness ===");

        SimpleClassifier model = new SimpleClassifier(10, 3);
        Predictor predictor = new Predictor(model).topK(3);

        Tensor input = Torch.rand(new int[]{1, 10});
        PredictionResult result = predictor.predict(input);

        float[] probs = result.getProbabilities();

        // All probabilities should be >= 0
        boolean allPositive = true;
        for (float p : probs) if (p < 0) allPositive = false;
        check("all probs >= 0", allPositive);

        // All probabilities should be <= 1
        boolean allLessOne = true;
        for (float p : probs) if (p > 1.0f + 1e-6) allLessOne = false;
        check("all probs <= 1", allLessOne);

        // Sum should be ~1
        check("probs sum ~1", Math.abs(sumArray(probs) - 1.0f) < 1e-4);
    }

    // ============================= HELPERS =============================

    static float sumArray(float[] arr) {
        float s = 0;
        for (float v : arr) s += v;
        return s;
    }

    // ============================= MAIN =============================

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════╗");
        System.out.println("║      PREDICT LIBRARY UNIT TESTS          ║");
        System.out.println("╚══════════════════════════════════════════╝");

        testPredictionResult();
        testPredictorSinglePredict();
        testPredictorBatchPredict();
        testPredictClass();
        testNoGradDuringPredict();
        testEvalMode();
        testImagePredictor();
        testImagePredictorFactory();
        testImagePredictorRgb();
        testPredictionPipeline();
        testPredictionPipelineImageMode();
        testTopKValues();
        testSoftmaxCorrectness();

        System.out.println("\n╔══════════════════════════════════════════╗");
        System.out.printf("║  Results: %d passed, %d failed            ║%n", pass, fail);
        System.out.println("╚══════════════════════════════════════════╝");

        if (fail > 0) System.exit(1);
    }
}
