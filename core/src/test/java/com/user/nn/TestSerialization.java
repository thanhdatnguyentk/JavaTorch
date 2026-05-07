package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.*;
import com.user.nn.containers.*;
import com.user.nn.activations.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

import java.io.IOException;
import java.nio.file.Path;

public class TestSerialization {

    @TempDir
    Path tempDir;

    @Test
    void testSaveLoadCPU() throws IOException {
        Sequential model = new Sequential(
            new Linear(10, 5, true),
            new ReLU(),
            new Linear(5, 2, true)
        );
        
        float[] initialData = captureParams(model);
        String path = tempDir.resolve("test_model_cpu.bin").toString();
        model.save(path);
        
        // Modify params slightly to ensure load actually does something
        Parameter p = model.parameters().get(0);
        p.getTensor().data[0] += 1.0f;
        
        model.load(path);
        float[] loadedData = captureParams(model);
        
        assertArrayEquals(initialData, loadedData, 1e-6f, "CPU Save/Load mismatch");
    }

    @Test
    @Tag("gpu")
    void testSaveLoadGPU() throws IOException {
        assumeTrue(CUDAOps.isAvailable(), "CUDA not available");

        Sequential model = new Sequential(
            new Linear(10, 5, true),
            new ReLU(),
            new Linear(5, 2, true)
        );
        model.to(Tensor.Device.GPU);
        
        float[] initialData = captureParams(model);
        String path = tempDir.resolve("test_model_gpu.bin").toString();
        model.save(path);
        
        // Modify to verify load
        model.toCPU();
        Parameter p = model.parameters().get(0);
        p.getTensor().data[0] += 1.0f;
        model.to(Tensor.Device.GPU);

        model.load(path);
        float[] loadedData = captureParams(model);
        
        assertArrayEquals(initialData, loadedData, 1e-6f, "GPU Save/Load mismatch");
    }

    private float[] captureParams(Module model) {
        int total = 0;
        for (Parameter p : model.parameters()) {
            total += p.getTensor().numel();
        }
        float[] out = new float[total];
        int offset = 0;
        for (Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            boolean wasGPU = t.isGPU();
            if (wasGPU) t.toCPU();
            System.arraycopy(t.data, 0, out, offset, t.data.length);
            offset += t.data.length;
            if (wasGPU) t.to(Tensor.Device.GPU);
        }
        return out;
    }
}
