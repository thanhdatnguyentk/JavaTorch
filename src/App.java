import com.user.nn.nn;
public class App {
    public static void main(String[] args) throws Exception {
        nn nn = new nn();

        // Prepare XOR dataset (4 samples)
        nn.Mat X = nn.mat_alloc(4, 3); // inputs with bias as last column
        X.es[0] = 0; X.es[1] = 0; X.es[2] = 1;
        X.es[3] = 0; X.es[4] = 1; X.es[5] = 1;
        X.es[6] = 1; X.es[7] = 0; X.es[8] = 1;
        X.es[9] = 1; X.es[10] = 1; X.es[11] = 1;

        nn.Mat Y = nn.mat_alloc(4, 1);
        Y.es[0] = 0; Y.es[1] = 1; Y.es[2] = 1; Y.es[3] = 0;

        // Network: 3 -> 2 -> 1 (includes bias in input)
        nn.Mat W1 = nn.mat_alloc(3, 2);
        nn.Mat W2 = nn.mat_alloc(2, 1);
        nn.mat_rand(W1, -1f, 1f);
        nn.mat_rand(W2, -1f, 1f);

        float lr = 0.5f;
        int epochs = 100000;

        nn.Mat hidden = nn.mat_alloc(4, 2);
        nn.Mat output = nn.mat_alloc(4, 1);

        for (int e = 0; e < epochs; e++) {
            // forward: hidden = sigmoid(X * W1)
            nn.mat_dot(hidden, X, W1);
            applySigmoid(hidden);

            // forward: output = sigmoid(hidden * W2)
            nn.mat_dot(output, hidden, W2);
            applySigmoid(output);

            // compute output error: delta_out = (output - Y) * output*(1-output)
            nn.Mat delta_out = nn.mat_alloc(output.rows, output.cols);
            // copy output into delta_out
            for (int i = 0; i < output.rows * output.cols; i++) delta_out.es[i] = output.es[i];
            nn.mat_sub(delta_out, Y); // delta_out = output - Y
            // multiply by sigmoid derivative
            for (int i = 0; i < delta_out.rows * delta_out.cols; i++) {
                float o = output.es[i];
                delta_out.es[i] = delta_out.es[i] * o * (1 - o);
            }

            // grad W2 = hidden^T * delta_out
            nn.Mat hidden_t = transpose(nn, hidden);
            nn.Mat gradW2 = nn.mat_alloc(hidden_t.rows, delta_out.cols);
            nn.mat_dot(gradW2, hidden_t, delta_out);

            // update W2: W2 -= lr * gradW2
            for (int i = 0; i < W2.rows * W2.cols; i++) {
                W2.es[i] -= lr * gradW2.es[i];
            }

            // backprop to hidden: delta_hidden = (delta_out * W2^T) .* hidden*(1-hidden)
            nn.Mat W2_t = transpose(nn, W2);
            nn.Mat delta_hidden = nn.mat_alloc(delta_out.rows, W2_t.cols);
            nn.mat_dot(delta_hidden, delta_out, W2_t);
            for (int i = 0; i < delta_hidden.rows * delta_hidden.cols; i++) {
                float h = hidden.es[i];
                delta_hidden.es[i] = delta_hidden.es[i] * h * (1 - h);
            }

            // grad W1 = X^T * delta_hidden
            nn.Mat X_t = transpose(nn, X);
            nn.Mat gradW1 = nn.mat_alloc(X_t.rows, delta_hidden.cols);
            nn.mat_dot(gradW1, X_t, delta_hidden);

            // update W1
            for (int i = 0; i < W1.rows * W1.cols; i++) {
                W1.es[i] -= lr * gradW1.es[i];
            }

            if (e % 1000 == 0) {
                float loss = mse(output, Y);
                System.out.println("Epoch " + e + " loss=" + loss);
            }
        }

        // Final predictions
        nn.mat_dot(hidden, X, W1);
        applySigmoid(hidden);
        nn.mat_dot(output, hidden, W2);
        applySigmoid(output);

        System.out.println("Final outputs:");
        for (int i = 0; i < output.rows; i++) {
            System.out.println(output.es[i]);
        }

    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + (float)Math.exp(-x));
    }

    static void applySigmoid(nn.Mat m) {
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = sigmoid(m.es[i]);
    }

    static nn.Mat transpose(nn nnInst, nn.Mat a) {
        nn.Mat t = nnInst.mat_alloc(a.cols, a.rows);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                t.es[j * t.cols + i] = a.es[i * a.cols + j];
            }
        }
        return t;
    }

    static float mse(nn.Mat out, nn.Mat y) {
        float s = 0f;
        int n = out.rows * out.cols;
        for (int i = 0; i < n; i++) {
            float d = out.es[i] - y.es[i];
            s += d * d;
        }
        return s / n;
    }
}
