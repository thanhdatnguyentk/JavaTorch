import com.user.nn.nn;
import com.user.nn.Tensor;
import com.user.nn.Torch;
import com.user.nn.optim;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.Collections;

public class TrainIris {
    public static void main(String[] args) throws Exception {
        nn lib = new nn();

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

        nn.Mat Xtrain = lib.mat_alloc(trainN, dim);
        nn.Mat Xtest = lib.mat_alloc(testN, dim);
        int[] Ytrain = new int[trainN];
        int[] Ytest = new int[testN];
        for (int i = 0; i < trainN; i++) {
            int id = idx[i];
            for (int j = 0; j < dim; j++)
                Xtrain.es[i * dim + j] = Xarr[id][j];
            Ytrain[i] = Y[id];
        }
        for (int i = 0; i < testN; i++) {
            int id = idx[trainN + i];
            for (int j = 0; j < dim; j++)
                Xtest.es[i * dim + j] = Xarr[id][j];
            Ytest[i] = Y[id];
        }

        // network: dim -> hidden -> classes
        int hidden = 10;
        int classes = 3;

        nn.Sequential model = new nn.Sequential();
        // Use true to include bias automatically
        model.add(new nn.Linear(lib, dim, hidden, true));
        model.add(new nn.ReLU());
        model.add(new nn.Linear(lib, hidden, classes, true));

        Tensor XtrainT = Torch.fromMat(Xtrain);

        // seed parameters
        long pSeed = 123L;
        for (nn.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            lib.mat_rand_seed(p.data, pSeed++, -0.5f, 0.5f);
            System.arraycopy(p.data.es, 0, t.data, 0, t.data.length);
            t.requires_grad = true;
        }

        float lr = 0.01f;
        int epochs = 20000;

        optim.Adam optimizer = new optim.Adam(model.parameters(), lr);

        for (int e = 0; e < epochs; e++) {
            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            Tensor logitsT = model.forward(XtrainT);
            Tensor lossT = nn.F.cross_entropy_tensor(logitsT, Ytrain);

            // Backward pass using autograd
            lossT.backward();

            // Optimizer step (Adam)
            optimizer.step();

            if (e % 100 == 0) {
                float acc = evaluate(model, Xtest, Ytest);
                System.out.println(String.format("Epoch %d loss=%.6f test_acc=%.4f", e, lossT.data[0], acc));
            }
        }

        float finalAcc = evaluate(model, Xtest, Ytest);
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

    static float evaluate(nn.Module model, nn.Mat X, int[] Y) {
        Tensor out = model.forward(Torch.fromMat(X));
        int correct = 0;
        int N = X.rows;
        int classes = out.shape[1];
        for (int i = 0; i < N; i++) {
            float max = Float.NEGATIVE_INFINITY;
            int best = 0;
            for (int j = 0; j < classes; j++) {
                if (out.data[i * classes + j] > max) {
                    max = out.data[i * classes + j];
                    best = j;
                }
            }
            if (best == Y[i])
                correct++;
        }
        return (float) correct / N;
    }
}
