package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class TestLinearReLU {
    public static void main(String[] args) throws Exception {

        // deterministic sizes
        int batch = 4;
        int inF = 5;
        int outF = 3;
        long seed = 12345L;

        NN.Mat input = NN.mat_alloc(batch, inF);
        NN.Mat weight = NN.mat_alloc(inF, outF);
        NN.Mat bias = NN.mat_alloc(1, outF);

        NN.mat_rand_seed(input, seed, -1f, 1f);
        NN.mat_rand_seed(weight, seed + 1, -0.5f, 0.5f);
        NN.mat_rand_seed(bias, seed + 2, -0.1f, 0.1f);

        String tmpDir = "tests/tmp";
        String inPath = tmpDir + "/input.csv";
        String wPath = tmpDir + "/weight.csv";
        String bPath = tmpDir + "/bias.csv";
        String pyOut = tmpDir + "/out_py.csv";

        try {
            NN.writeMatCSV(input, inPath);
            NN.writeMatCSV(weight, wPath);
            NN.writeMatCSV(bias, bPath);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
        }

        // run python reference
        ProcessBuilder pb = new ProcessBuilder("py", "-3.10", "tests/linear_relu_ref.py", inPath, wPath, bPath, pyOut);
        pb.inheritIO();
        Process p = pb.start();
        boolean finished = p.waitFor(10, TimeUnit.SECONDS);
        if (!finished) {
            p.destroyForcibly();
            System.err.println("Python reference timed out");
            System.exit(3);
        }
        if (p.exitValue() != 0) {
            System.err.println("Python reference failed with exit code " + p.exitValue());
            System.exit(4);
        }

        // run Java implementation: Linear + ReLU
        Linear linear = new Linear(inF, outF, true);
        // copy weights/bias from files into params to ensure same values used by both
        try {
            NN.Mat wpy = NN.readMatCSV(wPath);
            NN.Mat bpy = NN.readMatCSV(bPath);
            // override random init
            linear.weight = new Parameter(wpy);
            linear.bias = new Parameter(bpy);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(5);
        }

        NN.Mat jOut = linear.forward(input);
        ReLU relu = new ReLU();
        NN.Mat jOutRelu = relu.forward(jOut);

        // read python output
        NN.Mat pyMat = null;
        try {
            pyMat = NN.readMatCSV(pyOut);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(6);
        }

        // compare
        if (pyMat.rows != jOutRelu.rows || pyMat.cols != jOutRelu.cols) {
            System.err.println("Shape mismatch: py=" + pyMat.rows + "x" + pyMat.cols + " java=" + jOutRelu.rows + "x"
                    + jOutRelu.cols);
            System.exit(7);
        }

        float tol = 1e-4f;
        boolean ok = true;
        for (int i = 0; i < pyMat.rows * pyMat.cols; i++) {
            float a = pyMat.es[i];
            float b = jOutRelu.es[i];
            float diff = Math.abs(a - b);
            if (diff > tol) {
                System.err.println(String.format("Mismatch at %d: py=%f java=%f diff=%f", i, a, b, diff));
                ok = false;
            }
        }

        if (ok) {
            System.out.println("TEST PASSED: Linear+ReLU matches PyTorch reference within tol=" + tol);
            System.exit(0);
        } else {
            System.err.println("TEST FAILED");
            System.exit(8);
        }
    }
}
