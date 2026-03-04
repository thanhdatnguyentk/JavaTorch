import com.user.nn.nn;
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
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length < 5) continue;
                float[] f = new float[4];
                for (int i = 0; i < 4; i++) f[i] = Float.parseFloat(parts[i]);
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
        for (int i = 0; i < N; i++) { Xarr[i] = features.get(i); Y[i] = labels.get(i); }

        // normalize features per column (mean/std)
        int dim = 4;
        float[] mean = new float[dim];
        float[] std = new float[dim];
        for (int j = 0; j < dim; j++) {
            double s=0; for (int i=0;i<N;i++) s+=Xarr[i][j]; mean[j]=(float)(s/N);
            double ss=0; for (int i=0;i<N;i++){ double d = Xarr[i][j]-mean[j]; ss += d*d; } std[j]=(float)Math.sqrt(ss/N + 1e-8);
            for (int i=0;i<N;i++) Xarr[i][j] = (Xarr[i][j]-mean[j]) / std[j];
        }

        // shuffle and split
        int seed = 42; Random r = new Random(seed);
        Integer[] idx = new Integer[N]; for (int i=0;i<N;i++) idx[i]=i; List<Integer> idxList = Arrays.asList(idx); Collections.shuffle(idxList, r); idx = idxList.toArray(new Integer[0]);
        int trainN = (int)(N*0.8);
        int testN = N - trainN;

        nn.Mat Xtrain = lib.mat_alloc(trainN, dim);
        nn.Mat Xtest = lib.mat_alloc(testN, dim);
        int[] Ytrain = new int[trainN];
        int[] Ytest = new int[testN];
        for (int i=0;i<trainN;i++){
            int id = idx[i];
            for (int j=0;j<dim;j++) Xtrain.es[i*dim + j] = Xarr[id][j];
            Ytrain[i] = Y[id];
        }
        for (int i=0;i<testN;i++){
            int id = idx[trainN + i];
            for (int j=0;j<dim;j++) Xtest.es[i*dim + j] = Xarr[id][j];
            Ytest[i] = Y[id];
        }

        // network: dim -> hidden -> classes
        int hidden = 10; int classes = 3;
        nn.Mat W1 = lib.mat_alloc(dim, hidden);
        nn.Mat W2 = lib.mat_alloc(hidden, classes);
        lib.mat_rand_seed(W1, 123L, -0.5f, 0.5f);
        lib.mat_rand_seed(W2, 124L, -0.5f, 0.5f);

        float lr = 0.1f;
        int epochs = 2000;

        nn.Mat hiddenMat = lib.mat_alloc(trainN, hidden);
        nn.Mat logits = lib.mat_alloc(trainN, classes);

        for (int e = 0; e < epochs; e++) {
            // forward
            lib.mat_dot(hiddenMat, Xtrain, W1); // (trainN x hidden)
            applySigmoid(hiddenMat);
            lib.mat_dot(logits, hiddenMat, W2); // (trainN x classes)
            float loss = softmaxAndLoss(logits, Ytrain, null); // logits becomes probs

            // compute gradient at output: delta = probs - y_onehot
            nn.Mat delta = lib.mat_alloc(trainN, classes);
            for (int i=0;i<trainN;i++){
                for (int c=0;c<classes;c++){
                    float p = logits.es[i*classes + c];
                    float y = (Ytrain[i]==c)?1f:0f;
                    delta.es[i*classes + c] = p - y;
                }
            }

            // grad W2 = hidden^T * delta / trainN
            nn.Mat hidden_t = transpose(lib, hiddenMat);
            nn.Mat gradW2 = lib.mat_alloc(hidden_t.rows, delta.cols);
            lib.mat_dot(gradW2, hidden_t, delta);
            for (int i=0;i<gradW2.es.length;i++) gradW2.es[i] /= trainN;

            // update W2
            for (int i=0;i<W2.es.length;i++) W2.es[i] -= lr * gradW2.es[i];

            // backprop to hidden: delta_hidden = delta * W2^T .* sigmoid'(hidden)
            nn.Mat W2_t = transpose(lib, W2);
            nn.Mat delta_hidden = lib.mat_alloc(delta.rows, W2_t.cols);
            lib.mat_dot(delta_hidden, delta, W2_t);
            // multiply by sigmoid derivative
            for (int i=0;i<delta_hidden.es.length;i++){
                float h = hiddenMat.es[i]; delta_hidden.es[i] = delta_hidden.es[i] * h * (1 - h);
            }

            // grad W1 = Xtrain^T * delta_hidden / trainN
            nn.Mat X_t = transpose(lib, Xtrain);
            nn.Mat gradW1 = lib.mat_alloc(X_t.rows, delta_hidden.cols);
            lib.mat_dot(gradW1, X_t, delta_hidden);
            for (int i=0;i<gradW1.es.length;i++) gradW1.es[i] /= trainN;

            // update W1
            for (int i=0;i<W1.es.length;i++) W1.es[i] -= lr * gradW1.es[i];

            if (e % 100 == 0) {
                float acc = evaluate(lib, Xtest, Ytest, W1, W2);
                System.out.println(String.format("Epoch %d loss=%.6f test_acc=%.4f", e, loss, acc));
            }
        }

        float finalAcc = evaluate(lib, Xtest, Ytest, W1, W2);
        System.out.println("Final test accuracy=" + finalAcc);
    }

    static void downloadIfMissing(String urlStr, File dest) throws IOException {
        if (dest.exists() && dest.length() > 0) return;
        dest.getParentFile().mkdirs();
        System.out.println("Downloading iris dataset...");
        try (InputStream in = new URL(urlStr).openStream(); FileOutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[8192]; int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
        System.out.println("Saved to " + dest.getPath());
    }

    static int speciesToIndex(String s) {
        s = s.trim();
        if (s.startsWith("Iris-setosa")) return 0;
        if (s.startsWith("Iris-versicolor")) return 1;
        return 2;
    }

    static void applySigmoid(nn.Mat m) {
        for (int i = 0; i < m.es.length; i++) m.es[i] = 1.0f / (1.0f + (float)Math.exp(-m.es[i]));
    }

    static nn.Mat transpose(nn lib, nn.Mat a) {
        nn.Mat t = lib.mat_alloc(a.cols, a.rows);
        for (int i = 0; i < a.rows; i++) for (int j = 0; j < a.cols; j++) t.es[j * t.cols + i] = a.es[i * a.cols + j];
        return t;
    }

    // softmax in-place on logits, return average cross-entropy loss. If probsOut != null it's ignored (compat)
    static float softmaxAndLoss(nn.Mat logits, int[] targets, nn.Mat probsOut) {
        int batch = logits.rows; int classes = logits.cols;
        float total = 0f;
        for (int i=0;i<batch;i++){
            int base = i*classes;
            float max = Float.NEGATIVE_INFINITY;
            for (int j=0;j<classes;j++) if (logits.es[base+j] > max) max = logits.es[base+j];
            double sum = 0.0;
            for (int j=0;j<classes;j++) { double e = Math.exp(logits.es[base+j] - max); sum += e; logits.es[base+j] = (float)e; }
            double logsum = Math.log(sum) + max;
            int t = targets[i];
            double logit_target = Math.log(logits.es[base + t]) + max - logsum; // but simpler to compute prob after
            // now normalize
            for (int j=0;j<classes;j++) logits.es[base+j] = (float)(logits.es[base+j] / sum);
            float p_t = logits.es[base + t];
            total += - (float)Math.log(Math.max(p_t, 1e-15));
        }
        return total / batch;
    }

    static float evaluate(nn lib, nn.Mat X, int[] Y, nn.Mat W1, nn.Mat W2) {
        int N = X.rows; int dim = X.cols; int hidden = W1.cols; int classes = W2.cols;
        nn.Mat hiddenMat = lib.mat_alloc(N, hidden);
        lib.mat_dot(hiddenMat, X, W1);
        for (int i=0;i<hiddenMat.es.length;i++) hiddenMat.es[i] = 1.0f/(1.0f + (float)Math.exp(-hiddenMat.es[i]));
        nn.Mat logits = lib.mat_alloc(N, classes);
        lib.mat_dot(logits, hiddenMat, W2);
        // softmax
        int correct = 0;
        for (int i=0;i<N;i++){
            int base = i*classes; float max = Float.NEGATIVE_INFINITY; int argmax=0; double sum=0;
            for (int j=0;j<classes;j++){ if (logits.es[base+j] > max) max = logits.es[base+j]; }
            for (int j=0;j<classes;j++){ double e = Math.exp(logits.es[base+j] - max); sum += e; logits.es[base+j] = (float)e; }
            for (int j=0;j<classes;j++) logits.es[base+j] = (float)(logits.es[base+j]/sum);
            // pick argmax
            int best=0; float bestv=logits.es[base+0]; for (int j=1;j<classes;j++) if (logits.es[base+j] > bestv){ bestv=logits.es[base+j]; best=j; }
            if (best == Y[i]) correct++;
        }
        return (float)correct / N;
    }
}
