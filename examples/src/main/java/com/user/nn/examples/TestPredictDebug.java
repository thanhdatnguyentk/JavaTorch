package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.models.cv.LeNet;
import com.user.nn.dataloaders.MnistLoader;
import java.io.File;

public class TestPredictDebug {
    static final String DATA_DIR = "data/mnist/";

    public static void main(String[] args) throws Exception {
        System.out.println("Loading MNIST test data...");
        float[][] testImages = MnistLoader.loadImages(new File(DATA_DIR + "t10k-images-idx3-ubyte.gz"));
        int[] testLabels = MnistLoader.loadLabels(new File(DATA_DIR + "t10k-labels-idx1-ubyte.gz"));
        
        LeNet model = new LeNet();
        if (!new File("lenet_mnist.bin").exists()) {
            System.err.println("lenet_mnist.bin not found");
            return;
        }
        model.load("lenet_mnist.bin");
        System.out.println("Model loaded.");

        // First evaluate manually using CPU inference!
        model.eval();
        int correct = 0;
        int total = 100;
        for (int i = 0; i < total; i++) {
            Tensor input = Torch.tensor(testImages[i], 1, 1, 28, 28);
            Tensor logits = model.forward(input);
            int pred = 0;
            float max = Float.NEGATIVE_INFINITY;
            for(int j=0; j<10; j++) {
                if(logits.data[j] > max) { max = logits.data[j]; pred = j; }
            }
            if(pred == testLabels[i]) correct++;
        }
        System.out.println("CPU pure manual accuracy: " + correct + "/" + total);

        // Now move to GPU and test GPU inference!
        model.toGPU();
        int correctGpu = 0;
        for (int i = 0; i < total; i++) {
            Tensor input = Torch.tensor(testImages[i], 1, 1, 28, 28);
            input.toGPU(); // evaluate ON GPU
            Tensor logits = model.forward(input);
            logits.toCPU();
            int pred = 0;
            float max = Float.NEGATIVE_INFINITY;
            for(int j=0; j<10; j++) {
                if(logits.data[j] > max) { max = logits.data[j]; pred = j; }
            }
            if(pred == testLabels[i]) correctGpu++;
        }
        System.out.println("GPU manual accuracy: " + correctGpu + "/" + total);
        
        // Now use Predictor (which runs on CPU by default)
        com.user.nn.predict.ImagePredictor predictor = 
            com.user.nn.predict.ImagePredictor.forMnist(model);
        
        int correctPred = 0;
        for (int i = 0; i < total; i++) {
            com.user.nn.predict.PredictionResult res = predictor.predictFromPixels(testImages[i]);
            if (res.getPredictedClass() == testLabels[i]) correctPred++;
        }
        System.out.println("Predictor (CPU device) accuracy: " + correctPred + "/" + total);

        // Predictor with GPU
        predictor.device(Tensor.Device.GPU);
        int correctPredGpu = 0;
        for (int i = 0; i < total; i++) {
            com.user.nn.predict.PredictionResult res = predictor.predictFromPixels(testImages[i]);
            if (res.getPredictedClass() == testLabels[i]) correctPredGpu++;
        }
        System.out.println("Predictor (GPU device) accuracy: " + correctPredGpu + "/" + total);
    }
}
