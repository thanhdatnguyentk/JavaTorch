package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.pooling.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

public class TestConvPool {

    @TempDir
    Path tempDir;

    @Test
    @Tag("integration")
    @Tag("slow")
    void testConv2dAgainstPython() throws Exception {
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

        NN.mat_rand_seed(input, 11L, -1f, 1f);
        NN.mat_rand_seed(weight, 12L, -0.5f, 0.5f);
        NN.mat_rand_seed(bias, 13L, -0.1f, 0.1f);

        File inPath = tempDir.resolve("conv_in.csv").toFile();
        File wPath = tempDir.resolve("conv_w.csv").toFile();
        File bPath = tempDir.resolve("conv_b.csv").toFile();
        File pyOut = tempDir.resolve("conv_out_py.csv").toFile();

        NN.writeMatCSV(input, inPath.getAbsolutePath());
        NN.writeMatCSV(weight, wPath.getAbsolutePath());
        NN.writeMatCSV(bias, bPath.getAbsolutePath());

        // ProcessBuilder with py -3.10
        ProcessBuilder pb = new ProcessBuilder("py", "-3.10", "tests/conv_ref.py", 
                inPath.getAbsolutePath(), wPath.getAbsolutePath(), bPath.getAbsolutePath(),
                String.valueOf(batch), String.valueOf(inC), String.valueOf(inH), String.valueOf(inW),
                String.valueOf(kh), String.valueOf(kw), String.valueOf(stride), String.valueOf(pad), 
                pyOut.getAbsolutePath());
        pb.inheritIO();
        Process p = pb.start();
        boolean finished = p.waitFor(10, TimeUnit.SECONDS);
        assertTrue(finished, "Python reference timed out");
        assertEquals(0, p.exitValue(), "Python reference failed");

        // Run Java implementation
        Conv2d conv = new Conv2d(inC, outC, kh, kw, stride, stride, pad, pad, true);
        NN.Mat wpy = NN.readMatCSV(wPath.getAbsolutePath());
        NN.Mat bpy = NN.readMatCSV(bPath.getAbsolutePath());
        
        // Fix: Reshape weight to [outC, inC, kh, kw] for Conv2d
        Tensor tw = Torch.fromMat(wpy).reshape(outC, inC, kh, kw);
        conv.weight = new Parameter(tw);
        conv.bias = new Parameter(bpy);

        Tensor tin = Torch.fromMat(input);
        Tensor tout = conv.forward(tin);

        // Read python output
        NN.Mat pyMat = NN.readMatCSV(pyOut.getAbsolutePath());

        // Compare
        assertEquals(pyMat.rows, tout.shape[0]);
        int cSize = 1;
        for (int i = 1; i < tout.shape.length; i++) cSize *= tout.shape[i];
        assertEquals(pyMat.cols, cSize);

        float tol = 1e-4f;
        for (int i = 0; i < pyMat.rows * pyMat.cols; i++) {
            assertEquals(pyMat.es[i], tout.data[i], tol, "Mismatch at index " + i);
        }
    }

    @Test
    void testMaxPool2d() {
        int inC = 1;
        int inH = 4;
        int inW = 4;
        NN.Mat input = NN.mat_alloc(1, inC * inH * inW);
        Tensor tin = Torch.fromMat(input);

        MaxPool2d pool = new MaxPool2d(2, 2, 2, 2, 0, 0, inC, inH, inW);
        Tensor pOut = pool.forward(tin);

        int outH = (inH + 2 * 0 - 2) / 2 + 1;
        int outW = (inW + 2 * 0 - 2) / 2 + 1;
        int pCSize = 1;
        for (int i = 1; i < pOut.shape.length; i++) pCSize *= pOut.shape[i];
        assertEquals(inC * outH * outW, pCSize, "MaxPool shape mismatch");
    }
}
