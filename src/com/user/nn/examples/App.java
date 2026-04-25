package com.user.nn.examples;

import com.user.nn.core.*;

public class App {
    public static void main(String[] args) throws Exception {
        System.out.println("Hello, World!");

        // Prepare XOR dataset (4 samples)
        NN.Mat X = NN.mat_alloc(4, 3); // inputs with bias as last column
        X.es[0] = 0; X.es[1] = 0; X.es[2] = 1;
        X.es[3] = 0; X.es[4] = 1; X.es[5] = 1;
        X.es[6] = 1; X.es[7] = 0; X.es[8] = 1;
        X.es[9] = 1; X.es[10] = 1; X.es[11] = 1;

        NN.Mat Y = NN.mat_alloc(4, 1);
        Y.es[0] = 0; Y.es[1] = 1; Y.es[2] = 1; Y.es[3] = 0;

        // Network: 3 -> 2 -> 1 (includes bias in input)
        NN.Mat W1 = NN.mat_alloc(3, 2);
        NN.Mat W2 = NN.mat_alloc(2, 1);
        NN.mat_rand(W1, -1f, 1f);
        NN.mat_rand(W2, -1f, 1f);

        float lr = 0.5f;
        int epochs = SmokeTest.getEpochs(100000);

        NN.Mat hidden = NN.mat_alloc(4, 2);
        NN.Mat output = NN.mat_alloc(4, 1);

        for (int e = 0; e < epochs; e++) {
            // forward: hidden = sigmoid(X * W1)
            NN.mat_dot(hidden, X, W1);
            applySigmoid(hidden);

            // forward: output = sigmoid(hidden * W2)
            NN.mat_dot(output, hidden, W2);
            applySigmoid(output);

            // compute output error: delta_out = (output - Y) * output*(1-output)
            NN.Mat delta_out = NN.mat_alloc(output.rows, output.cols);
            // copy output into delta_out
            for (int i = 0; i < output.rows * output.cols; i++) delta_out.es[i] = output.es[i];
            NN.mat_sub(delta_out, Y); // delta_out = output - Y
            // multiply by sigmoid derivative
            for (int i = 0; i < delta_out.rows * delta_out.cols; i++) {
                float o = output.es[i];
                delta_out.es[i] = delta_out.es[i] * o * (1 - o);
            }

            // grad W2 = hidden^T * delta_out
            NN.Mat hidden_t = transpose(hidden);
            NN.Mat gradW2 = NN.mat_alloc(hidden_t.rows, delta_out.cols);
            NN.mat_dot(gradW2, hidden_t, delta_out);

            // update W2: W2 -= lr * gradW2
            for (int i = 0; i < W2.rows * W2.cols; i++) {
                W2.es[i] -= lr * gradW2.es[i];
            }

            // backprop to hidden: delta_hidden = (delta_out * W2^T) .* hidden*(1-hidden)
            NN.Mat W2_t = transpose(W2);
            NN.Mat delta_hidden = NN.mat_alloc(delta_out.rows, W2_t.cols);
            NN.mat_dot(delta_hidden, delta_out, W2_t);
            for (int i = 0; i < delta_hidden.rows * delta_hidden.cols; i++) {
                float h = hidden.es[i];
                delta_hidden.es[i] = delta_hidden.es[i] * h * (1 - h);
            }

            // grad W1 = X^T * delta_hidden
            NN.Mat X_t = transpose(X);
            NN.Mat gradW1 = NN.mat_alloc(X_t.rows, delta_hidden.cols);
            NN.mat_dot(gradW1, X_t, delta_hidden);

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
        NN.mat_dot(hidden, X, W1);
        applySigmoid(hidden);
        NN.mat_dot(output, hidden, W2);
        applySigmoid(output);

        System.out.println("Final outputs:");
        for (int i = 0; i < output.rows; i++) {
            System.out.println(output.es[i]);
        }

    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + (float)Math.exp(-x));
    }

    static void applySigmoid(NN.Mat m) {
        for (int i = 0; i < m.rows * m.cols; i++) m.es[i] = sigmoid(m.es[i]);
    }

    static NN.Mat transpose(NN.Mat a) {
        NN.Mat t = NN.mat_alloc(a.cols, a.rows);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                t.es[j * t.cols + i] = a.es[i * a.cols + j];
            }
        }
        return t;
    }

    static float mse(NN.Mat out, NN.Mat y) {
        float s = 0f;
        int n = out.rows * out.cols;
        for (int i = 0; i < n; i++) {
            float d = out.es[i] - y.es[i];
            s += d * d;
        }
        return s / n;
    }
}
