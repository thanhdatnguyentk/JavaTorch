package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.rnn.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestRNN {

    @Test
    void testRNNForwardBackward() {
        int inputSize = 4;
        int hiddenSize = 8;
        int seqLen = 5;
        int batch = 2;

        RNN rnn = new RNN(inputSize, hiddenSize, true, true);
        Tensor x = Torch.randn(new int[] { batch, seqLen, inputSize });
        x.requires_grad = true;

        Tensor out = rnn.forward(x);

        assertArrayEquals(new int[]{batch, seqLen, hiddenSize}, out.shape, "RNN output shape mismatch");

        // Use proper sumTensor for autograd
        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "RNN backward failed: x.grad is null");
    }

    @Test
    void testLSTMForwardBackward() {
        int inputSize = 4;
        int hiddenSize = 8;
        int seqLen = 5;
        int batch = 2;

        LSTM lstm = new LSTM(inputSize, hiddenSize, true, true);
        Tensor x = Torch.randn(new int[] { batch, seqLen, inputSize });
        x.requires_grad = true;

        Tensor out = lstm.forward(x);

        assertArrayEquals(new int[]{batch, seqLen, hiddenSize}, out.shape, "LSTM output shape mismatch");

        Tensor loss = Torch.sumTensor(out);
        loss.backward();

        assertNotNull(x.grad, "LSTM backward failed: x.grad is null");
    }
}
