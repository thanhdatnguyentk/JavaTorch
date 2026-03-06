package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.optim.*;
import com.user.nn.dataloaders.Data;
import com.user.nn.metrics.*;
import java.io.*;
import java.net.URL;
import java.util.*;

/**
 * TrainIris:
 * Demonstrates an end-to-end training loop for a 3-class classification
 * problem using a simple custom Dataset and DataLoader via `NN` and `optim`.
 */
public class TrainIris {
    public static void main(String[] args) throws Exception {
        NN lib = new NN();

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
        File csv = new File("tests/iris.csv");
        downloadIfMissing(url, csv);

        List<float[]> features = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csv))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;
                String[] parts = line.split(",");
                if (parts.length < 5)
                    continue;
                float[] f = new float[4];
                for (int i = 0; i < 4; i++)
                    f[i] = Float.parseFloat(parts[i]);
                int lab = speciesToIndex(parts[4]);
                features.add(f);
                labels.add(lab);
            }
        }

        int N = features.size();
        System.out.println("Loaded " + N + " samples");

        // build matrices
        float[][] Xarr = new float[N][];
        int[] Y = new int[N];
        for (int i = 0; i < N; i++) {
            Xarr[i] = features.get(i);
            Y[i] = labels.get(i);
        }

        // normalize features per column (mean/std)
        int dim = 4;
        float[] mean = new float[dim];
        float[] std = new float[dim];
        for (int j = 0; j < dim; j++) {
            double s = 0;
            for (int i = 0; i < N; i++)
                s += Xarr[i][j];
            mean[j] = (float) (s / N);
            double ss = 0;
            for (int i = 0; i < N; i++) {
                double d = Xarr[i][j] - mean[j];
                ss += d * d;
            }
            std[j] = (float) Math.sqrt(ss / N + 1e-8);
            for (int i = 0; i < N; i++)
                Xarr[i][j] = (Xarr[i][j] - mean[j]) / std[j];
        }

        // shuffle and split
        int seed = 42;
        Random r = new Random(seed);
        Integer[] idx = new Integer[N];
        for (int i = 0; i < N; i++)
            idx[i] = i;
        List<Integer> idxList = Arrays.asList(idx);
        Collections.shuffle(idxList, r);
        idx = idxList.toArray(new Integer[0]);
        int trainN = (int) (N * 0.8);
        int testN = N - trainN;

        NN.Mat XtrainMat = lib.mat_alloc(trainN, dim);
        NN.Mat XtestMat = lib.mat_alloc(testN, dim);
        int[] YtrainArr = new int[trainN];
        int[] YtestArr = new int[testN];
        for (int i = 0; i < trainN; i++) {
            int id = idx[i];
            for (int j = 0; j < dim; j++)
                XtrainMat.es[i * dim + j] = Xarr[id][j];
            YtrainArr[i] = Y[id];
        }
        for (int i = 0; i < testN; i++) {
            int id = idx[trainN + i];
            for (int j = 0; j < dim; j++)
                XtestMat.es[i * dim + j] = Xarr[id][j];
            YtestArr[i] = Y[id];
        }

        // Build Model
        NN.Sequential model = new NN.Sequential();
        model.add(new NN.Linear(lib, 4, 16, true));
        model.add(new NN.ReLU());
        model.add(new NN.Dropout(0.1f));
        model.add(new NN.Linear(lib, 16, 3, true));

        // Let's use Adam
        float lr = 0.05f;
        Optim.Adam optimizer = new Optim.Adam(model.parameters(), lr);
        
        // Move model to GPU
        model.toGPU();

        // Optional: Manual Train/Test split mechanism (80/20)
        int numSamples = N; // Use N from loaded data
        int numTrain = (int) (numSamples * 0.8);

        // Prepare data for Dataset
        float[][] data = new float[N][dim];
        int[] labelsData = new int[N];
        for (int i = 0; i < N; i++) {
            data[i] = Xarr[idx[i]]; // Use shuffled Xarr
            labelsData[i] = Y[idx[i]]; // Use shuffled Y
        }

        // Custom Train Dataset
        Data.Dataset trainDataset = new Data.Dataset() {
            @Override
            public int len() {
                return numTrain;
            }

            @Override
            public Tensor[] get(int index) {
                Tensor x = Torch.tensor(data[index], 1, 4);
                Tensor y = Torch.tensor(new float[] { labelsData[index] }, 1, 1);
                x = Torch.reshape(x, 4);
                y = Torch.reshape(y, 1);
                return new Tensor[] { x, y };
            }
        };

        // Create DataLoader for Training (batch size 16, shuffle=true, 2 workers)
        Data.DataLoader trainLoader = new Data.DataLoader(trainDataset, 16, true, 2);

        // seed parameters
        long pSeed = 123L;
        for (NN.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            lib.mat_rand_seed(p.data, pSeed++, -0.5f, 0.5f);
            System.arraycopy(p.data.es, 0, t.data, 0, t.data.length);
            t.requires_grad = true;
        }

        Accuracy accMetric = new Accuracy();
        int epochs = 2000;

        for (int e = 0; e < epochs; e++) {
            float epochLoss = 0.0f;
            int batchCount = 0;
            accMetric.reset();
            model.train();

            for (Tensor[] batch : trainLoader) {
                Tensor xBatch = batch[0]; // [batch_size, 4]
                Tensor yBatch = batch[1]; // [batch_size, 1]
                
                xBatch.toGPU();
                // yBatch stays on CPU for label indexing

                int bs = xBatch.shape[0];

                // Convert targets vector to int[] for cross entropy
                int[] batchLabels = new int[bs];
                for (int i = 0; i < bs; i++) {
                    batchLabels[i] = (int) yBatch.data[i];
                }

                optimizer.zero_grad();

                // Forward pass
                Tensor logits = model.forward(xBatch);

                // Loss
                Tensor loss = NN.F.cross_entropy_tensor(logits, batchLabels);
                epochLoss += loss.data[0];
                batchCount++;

                // Track accuracy
                accMetric.update(logits, batchLabels);

                // Backward pass using autograd
                loss.backward();

                // Optimizer step (Adam)
                optimizer.step();
                
                // Cleanup intermediate tensors on GPU
                xBatch.close();
                logits.close();
                loss.close();
            }

            if (e % 100 == 0) {
                float trainAcc = accMetric.compute();
                float testAcc = evaluate(model, XtestMat, YtestArr, accMetric);
                System.out.println(String.format("Epoch %d loss=%.6f train_acc=%.4f test_acc=%.4f", e, epochLoss / batchCount, trainAcc, testAcc));
            }
        }

        float finalAcc = evaluate(model, XtestMat, YtestArr, accMetric);
        System.out.println("Final test accuracy=" + finalAcc);
    }

    static void downloadIfMissing(String urlStr, File dest) throws IOException {
        if (dest.exists() && dest.length() > 0)
            return;
        dest.getParentFile().mkdirs();
        System.out.println("Downloading iris dataset...");
        try (InputStream in = new URL(urlStr).openStream(); FileOutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0)
                out.write(buf, 0, n);
        }
        System.out.println("Saved to " + dest.getPath());
    }

    static int speciesToIndex(String s) {
        s = s.trim();
        if (s.startsWith("Iris-setosa"))
            return 0;
        if (s.startsWith("Iris-versicolor"))
            return 1;
        return 2;
    }

    static float evaluate(NN.Module model, NN.Mat X, int[] Y, Accuracy metric) {
        model.eval();
        metric.reset();
        Tensor out = model.forward(Torch.fromMat(X));
        metric.update(out, Y);
        model.train();
        return metric.compute();
    }
}
