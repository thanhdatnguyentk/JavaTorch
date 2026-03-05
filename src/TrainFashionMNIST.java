import com.user.nn.*;
import java.io.*;
import java.util.*;

/**
 * Train an MLP on Fashion-MNIST using the framework's autograd, optim.Adam,
 * and cross_entropy_tensor loss.
 *
 * Fashion-MNIST: 28x28 grayscale, 10 classes, 60k train / 10k test.
 * Model: 784 → 256 (ReLU) → 128 (ReLU) → 10
 */
public class TrainFashionMNIST {

    static final String BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
    static final String DATA_DIR = "data/fashion-mnist/";

    public static void main(String[] args) throws Exception {
        // --- Download data ---
        String[] files = {
                "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        };
        for (String f : files)
            MnistLoader.downloadIfMissing(BASE_URL + f, new File(DATA_DIR + f));

        // --- Load data ---
        System.out.println("Loading data...");
        float[][] trainImages = MnistLoader.loadImages(new File(DATA_DIR + "train-images-idx3-ubyte.gz"));
        int[] trainLabels = MnistLoader.loadLabels(new File(DATA_DIR + "train-labels-idx1-ubyte.gz"));
        float[][] testImages = MnistLoader.loadImages(new File(DATA_DIR + "t10k-images-idx3-ubyte.gz"));
        int[] testLabels = MnistLoader.loadLabels(new File(DATA_DIR + "t10k-labels-idx1-ubyte.gz"));
        System.out.println("Train: " + trainImages.length + " images, Test: " + testImages.length + " images");

        // --- Build model ---
        nn lib = new nn();
        int inputDim = 784; // 28*28
        int hidden1 = 256;
        int hidden2 = 128;
        int numClasses = 10;

        nn.Sequential model = new nn.Sequential();
        model.add(new nn.Linear(lib, inputDim, hidden1, true));
        model.add(new nn.ReLU());
        model.add(new nn.Linear(lib, hidden1, hidden2, true));
        model.add(new nn.ReLU());
        model.add(new nn.Linear(lib, hidden2, numClasses, true));

        // Initialize parameters
        long seed = 42L;
        for (nn.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            // Kaiming-like init: scale by sqrt(2/fan_in)
            float scale = (float) Math.sqrt(2.0 / t.data.length);
            Random rng = new Random(seed++);
            for (int i = 0; i < t.data.length; i++) {
                t.data[i] = (float) (rng.nextGaussian() * scale);
            }
            t.requires_grad = true;
        }

        // --- Optimizer ---
        float lr = 0.001f;
        int epochs = 5;
        int batchSize = 64;
        optim.Adam optimizer = new optim.Adam(model.parameters(), lr);

        final int N = trainImages.length;

        // Define the Dataset for Fashion-MNIST
        data.Dataset trainDataset = new data.Dataset() {
            @Override
            public int len() {
                return N;
            }

            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(trainImages[index], 1, inputDim);
                Tensor y = Torch.tensor(new float[] { trainLabels[index] }, 1, 1);
                // We'll return just 1D tensors inside arrays, collateFn stacks them
                // Wait, collateFn does Torch.stack which adds a dimension.
                // So if we return [1, 784], it stacks to [bs, 1, 784].
                // Let's return flattened tensors.
                x = Torch.reshape(x, inputDim);
                y = Torch.reshape(y, 1);
                return new Tensor[] { x, y };
            }
        };

        // Create DataLoader with 4 worker threads
        data.DataLoader loader = new data.DataLoader(trainDataset, batchSize, true, 4);

        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int correct = 0;
            int numBatches = 0;

            for (Tensor[] batch : loader) {
                Tensor xBatch = batch[0];
                int bs = xBatch.shape[0];

                // Convert yBatch to int array for cross_entropy_tensor
                int[] batchLabels = new int[bs];
                for (int i = 0; i < bs; i++) {
                    batchLabels[i] = (int) batch[1].data[i];
                }

                // Forward
                optimizer.zero_grad();
                Tensor logits = model.forward(xBatch);
                Tensor loss = nn.F.cross_entropy_tensor(logits, batchLabels);

                // Backward + step
                loss.backward();
                optimizer.step();

                epochLoss += loss.data[0];
                numBatches++;

                // Compute train accuracy for this batch
                for (int i = 0; i < bs; i++) {
                    float maxVal = Float.NEGATIVE_INFINITY;
                    int pred = 0;
                    for (int j = 0; j < numClasses; j++) {
                        float v = logits.data[i * numClasses + j];
                        if (v > maxVal) {
                            maxVal = v;
                            pred = j;
                        }
                    }
                    if (pred == batchLabels[i])
                        correct++;
                }

                if (numBatches % 100 == 0) {
                    System.out.printf("  Epoch %d batch %d/%d  loss=%.4f%n",
                            epoch + 1, numBatches, N / batchSize, loss.data[0]);
                }
            }

            float trainAcc = (float) correct / (numBatches * batchSize);
            float avgLoss = epochLoss / numBatches;

            // --- Test accuracy ---
            float testAcc = evaluate(model, testImages, testLabels, numClasses);

            System.out.printf("Epoch %d/%d  avg_loss=%.4f  train_acc=%.4f  test_acc=%.4f%n",
                    epoch + 1, epochs, avgLoss, trainAcc, testAcc);
        }

        loader.shutdown();

        // Final test
        float finalAcc = evaluate(model, testImages, testLabels, numClasses);
        System.out.printf("%nFinal test accuracy: %.2f%%%n", finalAcc * 100);
    }

    static float evaluate(nn.Module model, float[][] images, int[] labels, int numClasses) {
        // Evaluate in batches to avoid OOM
        int N = images.length;
        int dim = images[0].length;
        int correct = 0;
        int evalBatch = 256;

        // Disable gradient tracking during evaluation
        for (int start = 0; start < N; start += evalBatch) {
            int bs = Math.min(evalBatch, N - start);
            float[] data = new float[bs * dim];
            for (int i = 0; i < bs; i++)
                System.arraycopy(images[start + i], 0, data, i * dim, dim);
            Tensor x = Torch.tensor(data, bs, dim);

            Tensor out;
            Torch.set_grad_enabled(false);
            out = model.forward(x);
            Torch.set_grad_enabled(true);

            for (int i = 0; i < bs; i++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                int pred = 0;
                for (int j = 0; j < numClasses; j++) {
                    float v = out.data[i * numClasses + j];
                    if (v > maxVal) {
                        maxVal = v;
                        pred = j;
                    }
                }
                if (pred == labels[start + i])
                    correct++;
            }
        }
        return (float) correct / N;
    }
}
