package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.activations.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

public class TestLinearReLU {

    @TempDir
    Path tempDir;

    @Test
    @Tag("integration")
    @Tag("slow")
    void testLinearReLUAgainstPython() throws Exception {
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

        File inPath = tempDir.resolve("input.csv").toFile();
        File wPath = tempDir.resolve("weight.csv").toFile();
        File bPath = tempDir.resolve("bias.csv").toFile();
        File pyOut = tempDir.resolve("out_py.csv").toFile();

        NN.writeMatCSV(input, inPath.getAbsolutePath());
        NN.writeMatCSV(weight, wPath.getAbsolutePath());
        NN.writeMatCSV(bias, bPath.getAbsolutePath());

        // Run python reference
        // Note: Using "py -3.10" as per user global rules
        ProcessBuilder pb = new ProcessBuilder("py", "-3.10", "tests/linear_relu_ref.py", 
                inPath.getAbsolutePath(), wPath.getAbsolutePath(), bPath.getAbsolutePath(), pyOut.getAbsolutePath());
        pb.inheritIO();
        Process p = pb.start();
        boolean finished = p.waitFor(10, TimeUnit.SECONDS);
        assertTrue(finished, "Python reference timed out");
        assertEquals(0, p.exitValue(), "Python reference failed");

        // Run Java implementation
        Linear linear = new Linear(inF, outF, true);
        NN.Mat wpy = NN.readMatCSV(wPath.getAbsolutePath());
        NN.Mat bpy = NN.readMatCSV(bPath.getAbsolutePath());
        linear.weight = new Parameter(wpy);
        linear.bias = new Parameter(bpy);

        NN.Mat jOut = linear.forward(input);
        ReLU relu = new ReLU();
        NN.Mat jOutRelu = relu.forward(jOut);

        // Read python output
        NN.Mat pyMat = NN.readMatCSV(pyOut.getAbsolutePath());

        // Compare
        assertEquals(pyMat.rows, jOutRelu.rows, "Rows mismatch");
        assertEquals(pyMat.cols, jOutRelu.cols, "Cols mismatch");

        float tol = 1e-4f;
        for (int i = 0; i < pyMat.rows * pyMat.cols; i++) {
            assertEquals(pyMat.es[i], jOutRelu.es[i], tol, "Value mismatch at index " + i);
        }
    }
}
