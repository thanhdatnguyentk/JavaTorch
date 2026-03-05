package com.user.nn;

import java.util.List;

public class TestRNN {
    public static void main(String[] args) {
        testRNNForwardBackward();
        testLSTMForwardBackward();
    }

    public static void testRNNForwardBackward() {
        System.out.println("Testing RNN Forward/Backward...");
        nn outer = new nn();
        int inputSize = 4;
        int hiddenSize = 8;
        int seqLen = 5;
        int batch = 2;

        nn.RNN rnn = new nn.RNN(outer, inputSize, hiddenSize, true, true);
        Tensor x = Torch.randn(new int[] { batch, seqLen, inputSize });
        x.requires_grad = true;

        Tensor out = rnn.forward(x);

        if (out.shape[0] != batch || out.shape[1] != seqLen || out.shape[2] != hiddenSize) {
            throw new RuntimeException("RNN output shape mismatch: " + java.util.Arrays.toString(out.shape));
        }

        // Loss = sum(out)
        float sum = 0;
        for (float f : out.data)
            sum += f;
        out.backward();

        if (x.grad == null) {
            throw new RuntimeException("RNN backward failed: x.grad is null");
        }

        System.out.println("RNN Test PASSED.");
    }

    public static void testLSTMForwardBackward() {
        System.out.println("Testing LSTM Forward/Backward...");
        nn outer = new nn();
        int inputSize = 4;
        int hiddenSize = 8;
        int seqLen = 5;
        int batch = 2;

        nn.LSTM lstm = new nn.LSTM(outer, inputSize, hiddenSize, true, true);
        Tensor x = Torch.randn(new int[] { batch, seqLen, inputSize });
        x.requires_grad = true;

        Tensor out = lstm.forward(x);

        if (out.shape[0] != batch || out.shape[1] != seqLen || out.shape[2] != hiddenSize) {
            throw new RuntimeException("LSTM output shape mismatch: " + java.util.Arrays.toString(out.shape));
        }

        out.backward();

        if (x.grad == null) {
            throw new RuntimeException("LSTM backward failed: x.grad is null");
        }

        System.out.println("LSTM Test PASSED.");
    }
}
