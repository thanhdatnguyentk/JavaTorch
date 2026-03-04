package com.user.nn;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class TestLinearReLU {
    public static void main(String[] args) throws Exception {
        nn lib = new nn();

        // deterministic sizes
        int batch = 4;
        int inF = 5;
        int outF = 3;
        long seed = 12345L;

        nn.Mat input = lib.mat_alloc(batch, inF);
        nn.Mat weight = lib.mat_alloc(inF, outF);
        nn.Mat bias = lib.mat_alloc(1, outF);

        lib.mat_rand_seed(input, seed, -1f, 1f);
        lib.mat_rand_seed(weight, seed + 1, -0.5f, 0.5f);
        lib.mat_rand_seed(bias, seed + 2, -0.1f, 0.1f);

        String tmpDir = "tests/tmp";
        String inPath = tmpDir + "/input.csv";
        String wPath = tmpDir + "/weight.csv";
        String bPath = tmpDir + "/bias.csv";
        String pyOut = tmpDir + "/out_py.csv";

        try {
            lib.writeMatCSV(input, inPath);
            lib.writeMatCSV(weight, wPath);
            lib.writeMatCSV(bias, bPath);
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
        nn.Linear linear = new nn.Linear(lib, inF, outF, true);
        // copy weights/bias from files into params to ensure same values used by both
        try {
            nn.Mat wpy = lib.readMatCSV(wPath);
            nn.Mat bpy = lib.readMatCSV(bPath);
            // override random init
            linear.weight.data = wpy;
            linear.bias.data = bpy;
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(5);
        }

        nn.Mat jOut = linear.forward(input);
        nn.ReLU relu = new nn.ReLU();
        nn.Mat jOutRelu = relu.forward(jOut);

        // read python output
        nn.Mat pyMat = null;
        try {
            pyMat = lib.readMatCSV(pyOut);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(6);
        }

        // compare
        if (pyMat.rows != jOutRelu.rows || pyMat.cols != jOutRelu.cols) {
            System.err.println("Shape mismatch: py=" + pyMat.rows + "x" + pyMat.cols + " java=" + jOutRelu.rows + "x" + jOutRelu.cols);
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
