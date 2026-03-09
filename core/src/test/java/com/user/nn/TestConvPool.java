package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.pooling.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class TestConvPool {
    public static void main(String[] args) throws Exception {
        int batch = 1;
        int inC = 1;
        int inH = 4;
        int inW = 4;
        int kh = 3;
        int kw = 3;
        int stride = 1;
        int pad = 0;
        int outC = 1;

        NN.Mat input = NN.mat_alloc(batch, inC * inH * inW);
        NN.Mat weight = NN.mat_alloc(inC * kh * kw, outC);
        NN.Mat bias = NN.mat_alloc(1, outC);

        // deterministic values
        NN.mat_rand_seed(input, 11L, -1f, 1f);
        NN.mat_rand_seed(weight, 12L, -0.5f, 0.5f);
        NN.mat_rand_seed(bias, 13L, -0.1f, 0.1f);

        String tmp = "tests/tmp";
        String inPath = tmp + "/conv_in.csv";
        String wPath = tmp + "/conv_w.csv";
        String bPath = tmp + "/conv_b.csv";
        String pyOut = tmp + "/conv_out_py.csv";

        try {
            NN.writeMatCSV(input, inPath);
            NN.writeMatCSV(weight, wPath);
            NN.writeMatCSV(bias, bPath);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
        }

        ProcessBuilder pb = new ProcessBuilder("py", "-3.10", "tests/conv_ref.py", inPath, wPath, bPath,
                String.valueOf(batch), String.valueOf(inC), String.valueOf(inH), String.valueOf(inW),
                String.valueOf(kh), String.valueOf(kw), String.valueOf(stride), String.valueOf(pad), pyOut);
        pb.inheritIO();
        Process p = pb.start();
        boolean finished = p.waitFor(10, TimeUnit.SECONDS);
        if (!finished) {
            p.destroyForcibly();
            System.err.println("Python conv timed out");
            System.exit(3);
        }
        if (p.exitValue() != 0) {
            System.err.println("Python conv failed");
            System.exit(4);
        }

        // run java conv
        Conv2d conv = new Conv2d(inC, outC, kh, kw, inH, inW, stride, pad, true);
        try {
            NN.Mat wpy = NN.readMatCSV(wPath);
            NN.Mat bpy = NN.readMatCSV(bPath);
            conv.weight = new Parameter(wpy);
            conv.bias = new Parameter(bpy);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(5);
        }

        Tensor tin = Torch.fromMat(input);
        Tensor tout = conv.forward(tin);
        // Convert back to Mat for comparison
        NN.Mat jOut = new NN.Mat();
        jOut.rows = tout.shape[0];
        int cSize = 1;
        for (int i = 1; i < tout.shape.length; i++) cSize *= tout.shape[i];
        jOut.cols = cSize;
        jOut.es = new float[jOut.rows * jOut.cols];
        System.arraycopy(tout.data, 0, jOut.es, 0, jOut.es.length);

        NN.Mat pyMat = null;
        try {
            pyMat = NN.readMatCSV(pyOut);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(6);
        }

        if (pyMat.rows != jOut.rows || pyMat.cols != jOut.cols) {
            System.err.println("conv shape mismatch");
            System.exit(7);
        }

        float tol = 1e-4f;
        boolean ok = true;
        for (int i = 0; i < pyMat.rows * pyMat.cols; i++) {
            if (Math.abs(pyMat.es[i] - jOut.es[i]) > tol) {
                System.err.println("conv mismatch idx=" + i + " py=" + pyMat.es[i] + " j=" + jOut.es[i]);
                ok = false;
            }
        }
        if (!ok) {
            System.err.println("TEST FAILED: Conv2d");
            System.exit(8);
        }

        // test MaxPool2d
        MaxPool2d pool = new MaxPool2d(2, 2, 2, 2, 0, 0, inC, inH, inW);
        Tensor pOut = pool.forward(tin);
        // check shape
        int outH = (inH + 2 * 0 - 2) / 2 + 1;
        int outW = (inW + 2 * 0 - 2) / 2 + 1;
        int pCSize = 1;
        for (int i = 1; i < pOut.shape.length; i++) pCSize *= pOut.shape[i];
        if (pCSize != inC * outH * outW) {
            System.err.println("MaxPool shape mismatch: " + pCSize + " vs " + (inC * outH * outW));
            System.exit(9);
        }

        System.out.println("TEST PASSED: ConvPool");
    }
}
