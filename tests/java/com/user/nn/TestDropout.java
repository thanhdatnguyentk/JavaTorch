package com.user.nn;
import com.user.nn.core.*;
import java.util.Arrays;

public class TestDropout {
    public static void main(String[] args) {
        testDropoutEval();
        testDropoutTrain();
        testDropoutGrad();
        testDropoutFunctional();
        System.out.println("TestDropout PASSED!");
    }

    private static void check(String name, boolean cond) {
        if (cond) {
            System.out.println("  PASS: " + name);
        } else {
            System.err.println("  FAIL: " + name);
            System.exit(1);
        }
    }

    private static void testDropoutEval() {
        System.out.println("Testing Dropout Eval mode...");
        NN.Dropout dropout = new NN.Dropout(0.5f);
        dropout.eval();
        
        Tensor x = Torch.ones(10, 10);
        Tensor out = dropout.forward(x);
        
        boolean identical = true;
        for(int i=0; i<x.data.length; i++) {
            if (out.data[i] != x.data[i]) {
                identical = false;
                break;
            }
        }
        check("Dropout in eval mode is identity", identical);
    }

    private static void testDropoutTrain() {
        System.out.println("Testing Dropout Train mode (scaling)...");
        NN.Dropout dropout = new NN.Dropout(0.5f);
        dropout.train(); // default is true
        
        // Large tensor to get stable statistics
        int n = 10000;
        Tensor x = Torch.ones(n);
        Tensor out = dropout.forward(x);
        
        int zeroCount = 0;
        float sum = 0;
        for(float v : out.data) {
            if (v == 0) zeroCount++;
            else {
                if (Math.abs(v - 2.0f) > 1e-6f) {
                    System.err.println("Expected scaled value 2.0, got " + v);
                    System.exit(1);
                }
            }
            sum += v;
        }
        
        float zeroRate = (float)zeroCount / n;
        System.out.println("  Zero rate: " + zeroRate);
        check("Zero rate roughly 0.5", zeroRate > 0.4 && zeroRate < 0.6);
        
        float avg = sum / n;
        System.out.println("  Average after dropout: " + avg);
        check("Average roughly 1.0 (inverted scaling check)", avg > 0.9 && avg < 1.1);
    }

    private static void testDropoutGrad() {
        System.out.println("Testing Dropout Gradient...");
        NN.Dropout dropout = new NN.Dropout(0.5f);
        dropout.train();
        
        Tensor x = Torch.ones(100);
        x.requires_grad = true;
        Tensor out = dropout.forward(x);
        
        Tensor loss = Torch.sumTensor(out);
        loss.backward();
        
        boolean validGrad = true;
        for(int i=0; i<x.data.length; i++) {
            if (out.data[i] == 0) {
                if (x.grad.data[i] != 0) validGrad = false;
            } else {
                // scale = 1 / (1 - 0.5) = 2.0
                if (Math.abs(x.grad.data[i] - 2.0f) > 1e-6f) validGrad = false;
            }
        }
        check("Dropout gradient follows mask and scaling", validGrad);
    }

    private static void testDropoutFunctional() {
        System.out.println("Testing NN.F.dropout...");
        Tensor x = Torch.ones(10, 10);
        
        // Test training mode
        Tensor outTrain = NN.F.dropout(x, 0.5f, true);
        float sumTrain = 0;
        for(float v : outTrain.data) sumTrain += v;
        check("F.dropout training active (sum not 100 or scale 2)", sumTrain != 100.0f || outTrain.data[0] == 2.0f || outTrain.data[0] == 0.0f);
        
        // Test eval mode
        Tensor outEval = NN.F.dropout(x, 0.5f, false);
        float sumEval = 0;
        for(float v : outEval.data) sumEval += v;
        check("F.dropout eval mode identity (sum=100)", sumEval == 100.0f);
    }
}
