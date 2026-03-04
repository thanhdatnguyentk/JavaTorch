package com.user.nn;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class TestConvPool {
    public static void main(String[] args) throws Exception {
        nn lib = new nn();
        int batch = 1;
        int inC = 1; int inH = 4; int inW = 4;
        int kh = 3; int kw = 3; int stride = 1; int pad = 0;
        int outC = 1;

        nn.Mat input = lib.mat_alloc(batch, inC * inH * inW);
        nn.Mat weight = lib.mat_alloc(inC * kh * kw, outC);
        nn.Mat bias = lib.mat_alloc(1, outC);

        // deterministic values
        lib.mat_rand_seed(input, 11L, -1f, 1f);
        lib.mat_rand_seed(weight, 12L, -0.5f, 0.5f);
        lib.mat_rand_seed(bias, 13L, -0.1f, 0.1f);

        String tmp = "tests/tmp";
        String inPath = tmp + "/conv_in.csv";
        String wPath = tmp + "/conv_w.csv";
        String bPath = tmp + "/conv_b.csv";
        String pyOut = tmp + "/conv_out_py.csv";

        try {
            lib.writeMatCSV(input, inPath);
            lib.writeMatCSV(weight, wPath);
            lib.writeMatCSV(bias, bPath);
        } catch (IOException e) { e.printStackTrace(); System.exit(2); }

        ProcessBuilder pb = new ProcessBuilder("py", "-3.10", "tests/conv_ref.py", inPath, wPath, bPath, String.valueOf(batch), String.valueOf(inC), String.valueOf(inH), String.valueOf(inW), String.valueOf(kh), String.valueOf(kw), String.valueOf(stride), String.valueOf(pad), pyOut);
        pb.inheritIO();
        Process p = pb.start();
        boolean finished = p.waitFor(10, TimeUnit.SECONDS);
        if (!finished) { p.destroyForcibly(); System.err.println("Python conv timed out"); System.exit(3); }
        if (p.exitValue() != 0) { System.err.println("Python conv failed"); System.exit(4); }

        // run java conv
        nn.Conv2d conv = new nn.Conv2d(inC, outC, kh, kw, inH, inW, stride, pad, lib, true);
        try {
            nn.Mat wpy = lib.readMatCSV(wPath);
            nn.Mat bpy = lib.readMatCSV(bPath);
            conv.weight.data = wpy;
            conv.bias.data = bpy;
        } catch (IOException e) { e.printStackTrace(); System.exit(5); }

        nn.Mat jOut = conv.forward(input);

        nn.Mat pyMat = null;
        try { pyMat = lib.readMatCSV(pyOut); } catch (IOException e) { e.printStackTrace(); System.exit(6); }

        if (pyMat.rows != jOut.rows || pyMat.cols != jOut.cols) { System.err.println("conv shape mismatch"); System.exit(7); }

        float tol = 1e-4f; boolean ok=true;
        for (int i=0;i<pyMat.rows*pyMat.cols;i++) {
            if (Math.abs(pyMat.es[i] - jOut.es[i]) > tol) { System.err.println("conv mismatch idx="+i+" py="+pyMat.es[i]+" j="+jOut.es[i]); ok=false; }
        }
        if (!ok) { System.err.println("TEST FAILED: Conv2d"); System.exit(8); }

        // test MaxPool2d
        nn.MaxPool2d pool = new nn.MaxPool2d(2,2,2,2,0,0,inC,inH,inW);
        nn.Mat pOut = pool.forward(input);
        // compute reference numpy maxpool quickly using conv_ref trick: we will not run python again; check shape
        int outH = (inH + 2*0 - 2)/2 +1;
        int outW = (inW + 2*0 - 2)/2 +1;
        if (pOut.cols != inC * outH * outW) { System.err.println("MaxPool shape mismatch"); System.exit(9); }

        System.out.println("TEST PASSED: ConvPool");
    }
}

